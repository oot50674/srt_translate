from __future__ import annotations

import os
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from werkzeug.datastructures import FileStorage

from constants import BASE_DIR
from module.video_split import (
    DEFAULT_STORAGE_KEY,
    SegmentMetadata,
    split_video_by_minutes,
)
from module.Whisper_util import WhisperUtil
from module import srt_module
from module.subtitle_sync import (
    SyncConfig,
    cleanup_subtitles,
    sync_subtitles,
    export_srt,
)
from module.hallucination_filter import fix_repetitive_hallucinations


WHISPER_BATCH_ROOT = os.path.join(BASE_DIR, "whisper_batches")
os.makedirs(WHISPER_BATCH_ROOT, exist_ok=True)

_BATCHES: Dict[str, "WhisperBatch"] = {}
_LOCK = threading.RLock()
_WHISPER_LOCK = threading.Lock()
_WHISPER_INSTANCE: Optional[WhisperUtil] = None
_ACTIVE_BATCHES = 0


def _get_whisper() -> WhisperUtil:
    global _WHISPER_INSTANCE
    with _WHISPER_LOCK:
        if _WHISPER_INSTANCE is None:
            _WHISPER_INSTANCE = WhisperUtil()
        return _WHISPER_INSTANCE


def _increment_active_batches() -> None:
    global _ACTIVE_BATCHES
    with _LOCK:
        _ACTIVE_BATCHES += 1


def _unload_whisper() -> None:
    global _WHISPER_INSTANCE
    with _WHISPER_LOCK:
        if _WHISPER_INSTANCE is None:
            return
        try:
            _WHISPER_INSTANCE.unload_model()
        except Exception:
            pass
        _WHISPER_INSTANCE = None


def _decrement_active_batches() -> None:
    global _ACTIVE_BATCHES
    with _LOCK:
        if _ACTIVE_BATCHES > 0:
            _ACTIVE_BATCHES -= 1
        if _ACTIVE_BATCHES == 0:
            _unload_whisper()


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_INVALID_CHARS = set('<>:"/\\|?*\0')
_YOUTUBE_HOST_SUFFIXES = ("youtube.com", "youtu.be")


def _sanitize_filename(original: str, fallback: str) -> str:
    name = os.path.basename(original or "").strip()
    if not name:
        name = fallback
    name = unicodedata.normalize("NFC", name)
    safe_chars: List[str] = []
    for ch in name:
        if ch in _INVALID_CHARS or ord(ch) < 32:
            safe_chars.append("_")
        else:
            safe_chars.append(ch)
    sanitized = "".join(safe_chars).strip().lstrip(".")
    if not sanitized:
        sanitized = fallback
    return sanitized[:255]


def _is_youtube_host(host: str) -> bool:
    if not host:
        return False
    lowered = host.lower()
    return any(lowered == suffix or lowered.endswith(f".{suffix}") for suffix in _YOUTUBE_HOST_SUFFIXES)


def _normalize_youtube_urls(urls: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if not _is_youtube_host(parsed.netloc):
            continue
        normalized_url = parsed.geturl()
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        normalized.append(normalized_url)
    return normalized


def _download_youtube_video(url: str, output_dir: str) -> str:
    _ensure_dir(output_dir)
    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    base, _ = os.path.splitext(path)
    mp4_path = base + ".mp4"
    if os.path.exists(mp4_path):
        return mp4_path
    return path


def _youtube_display_name(url: str, fallback: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return fallback
    if not _is_youtube_host(parsed.netloc):
        return fallback
    video_id = ""
    if "youtu.be" in parsed.netloc.lower():
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        query = parse_qs(parsed.query or "")
        if "v" in query and query["v"]:
            video_id = query["v"][0]
    if video_id:
        return f"YouTube - {video_id}"
    return fallback


def _format_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(entries: List[Dict[str, Any]], dest: str) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        start_ts = _format_timestamp(entry["start"])
        end_ts = _format_timestamp(entry["end"])
        text = entry.get("text", "").strip() or "[BLANK]"
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    content = "\n".join(lines).strip() + "\n"
    with open(dest, "w", encoding="utf-8") as fp:
        fp.write(content)
    return dest


def _save_uploaded_file(uploaded_file: FileStorage, dest_dir: str) -> str:
    fallback = f"upload_{uuid.uuid4().hex}.bin"
    filename = _sanitize_filename(uploaded_file.filename or "", fallback)
    target_dir = _ensure_dir(dest_dir)
    dest = os.path.join(target_dir, filename)
    uploaded_file.save(dest)
    return dest


@dataclass
class WhisperBatchItem:
    item_id: str
    file_name: str
    source_path: str
    segments_dir: str
    output_dir: str
    youtube_url: Optional[str] = None
    downloads_dir: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    transcript_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    download_ready: bool = False
    apply_vad_cleanup: bool = True
    apply_vad_sync: bool = True
    speech_timeline: List[Dict[str, float]] = field(default_factory=list)
    silent_entries_removed: int = 0

    def to_dict(self, batch_id: str) -> Dict[str, Any]:
        download_url = None
        if self.transcript_path and os.path.isfile(self.transcript_path):
            download_url = f"/api/whisper/batch/{batch_id}/items/{self.item_id}/download"
        return {
            "item_id": self.item_id,
            "file_name": self.file_name,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "transcript_ready": bool(download_url and self.download_ready),
            "download_url": download_url,
            "updated_at": self.updated_at,
            "apply_vad_cleanup": self.apply_vad_cleanup,
            "apply_vad_sync": self.apply_vad_sync,
            "silent_entries_removed": self.silent_entries_removed,
        }


@dataclass
class WhisperBatch:
    batch_id: str
    chunk_seconds: float
    base_dir: str
    apply_vad_cleanup: bool = True
    apply_vad_sync: bool = True
    items: List[WhisperBatchItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "pending"
    _worker_started: bool = field(default=False, init=False, repr=False)

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        entries = [item.to_dict(self.batch_id) for item in self.items]
        total = len(entries) or 1
        completed = sum(1 for item in self.items if item.status == "completed")
        failed = sum(1 for item in self.items if item.status == "failed")
        running = sum(1 for item in self.items if item.status == "running")
        overall = sum(item.progress for item in self.items) / total
        return {
            "batch_id": self.batch_id,
            "chunk_seconds": self.chunk_seconds,
            "apply_vad_cleanup": self.apply_vad_cleanup,
            "apply_vad_sync": self.apply_vad_sync,
            "status": self.status,
            "items": entries,
            "total_items": len(entries),
            "completed_items": completed,
            "failed_items": failed,
            "running_items": running,
            "overall_progress": overall,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def start_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        thread = threading.Thread(target=_run_batch, args=(self,), daemon=True)
        thread.start()


def _filter_uploads(files: Iterable[Optional[FileStorage]]) -> List[FileStorage]:
    uploads: List[FileStorage] = []
    for uploaded in files:
        if uploaded and uploaded.filename:
            uploads.append(uploaded)
    return uploads


def _update_item(
    batch: WhisperBatch,
    item: WhisperBatchItem,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    if status:
        item.status = status
    if progress is not None:
        item.progress = max(0.0, min(1.0, progress))
    if message is not None:
        item.message = message
    if error is not None:
        item.error = error
    item.updated_at = time.time()
    batch.touch()


def _ensure_source_path(batch: WhisperBatch, item: WhisperBatchItem) -> str:
    if item.source_path and os.path.isfile(item.source_path):
        return item.source_path
    if not item.youtube_url:
        raise FileNotFoundError("원본 파일을 찾을 수 없습니다.")
    _update_item(batch, item, message="YouTube 영상 다운로드 중입니다...")
    try:
        dest = _download_youtube_video(item.youtube_url, item.downloads_dir or item.segments_dir)
    except Exception as exc:
        raise RuntimeError(f"YouTube 다운로드 실패: {exc}") from exc
    item.source_path = dest
    try:
        if not item.file_name or item.file_name == item.youtube_url:
            item.file_name = os.path.basename(dest)
    except Exception:
        item.file_name = os.path.basename(dest)
    return dest


def _transcribe_segments(
    batch: WhisperBatch,
    item: WhisperBatchItem,
    segments: List[SegmentMetadata],
) -> str:
    whisper = _get_whisper()
    entries: List[Dict[str, Any]] = []
    total = len(segments) or 1
    for idx, segment in enumerate(segments, start=1):
        _update_item(
            batch,
            item,
            progress=(idx - 1) / total,
            message=f"{idx}/{total}개 세그먼트 전사 중...",
        )
        result = whisper.transcribe_audio(segment.file_path, show_progress=False)
        for snippet in result.get("segments", []) or []:
            text = str(snippet.get("text", "")).strip()
            if not text:
                continue
            start = segment.start_time + float(snippet.get("start", 0.0))
            end = segment.start_time + float(snippet.get("end", 0.0))
            entries.append(
                {
                    "start": max(0.0, start),
                    "end": max(0.0, end),
                    "text": text,
                }
            )
    if not entries:
        entries.append({"start": 0.0, "end": 0.5, "text": "[NO SPEECH DETECTED]"})
    entries.sort(key=lambda entry: entry["start"])
    try:
        entries, hallucination_stats = fix_repetitive_hallucinations(
            entries,
            item.source_path,
            whisper=whisper,
        )
        flagged = hallucination_stats.get("flagged", 0)
        retranscribed = hallucination_stats.get("retranscribed", 0)
        if flagged or retranscribed:
            _update_item(
                batch,
                item,
                message=f"환각 의심 자막 {retranscribed}건 재전사, {flagged}건 표시 처리 중...",
            )
    except Exception as exc:
        _update_item(batch, item, message=f"환각 반복 검사 건너뜀: {exc}")
    transcript_dir = _ensure_dir(item.output_dir)
    base_name = os.path.splitext(item.file_name)[0] or item.item_id
    target_name = _sanitize_filename(f"{base_name}.srt", f"{item.item_id}.srt")
    dest = os.path.join(transcript_dir, target_name)
    _write_srt(entries, dest)
    return dest


def _apply_vad_sync(batch: WhisperBatch, item: WhisperBatchItem, *, apply_cleanup: bool) -> bool:
    if not item.transcript_path or not os.path.isfile(item.transcript_path):
        item.download_ready = False
        return False
    item.download_ready = False
    _update_item(batch, item, message="VAD 싱크 보정 중입니다...")
    try:
        subtitles = srt_module.read_srt(item.transcript_path)
        config = SyncConfig(chunk_mode="individual", remove_silent_entries=apply_cleanup)
        segments_source = item.speech_timeline if item.speech_timeline else None
        synced, stats = sync_subtitles(
            subtitles,
            item.source_path,
            config,
            precomputed_segments=segments_source,
        )
        if stats.get("status") != "success":
            _update_item(
                batch,
                item,
                message=f"VAD 보정 건너뜀 ({stats.get('status')})",
            )
            item.silent_entries_removed = 0
            return False
        synced_content = export_srt(synced)
        with open(item.transcript_path, "w", encoding="utf-8") as fp:
            fp.write(synced_content)
        removed_silent = stats.get("silent_entries_removed", 0)
        item.silent_entries_removed = removed_silent if apply_cleanup else 0
        summary_msg = f"VAD 싱크 보정을 완료했습니다. (무음 엔트리 {item.silent_entries_removed}개 제거)"
        _update_item(batch, item, message=summary_msg)
        return True
    except Exception as exc:
        _update_item(
            batch,
            item,
            message=f"VAD 싱크 보정 실패: {exc}",
            error=str(exc),
        )
        item.silent_entries_removed = 0
        return False
    finally:
        item.download_ready = bool(item.transcript_path and os.path.isfile(item.transcript_path))


def _apply_vad_cleanup_only(batch: WhisperBatch, item: WhisperBatchItem) -> bool:
    if not item.transcript_path or not os.path.isfile(item.transcript_path):
        item.download_ready = False
        return False
    item.download_ready = False
    _update_item(batch, item, message="무음 환각 자막 제거 중입니다...")
    try:
        subtitles = srt_module.read_srt(item.transcript_path)
        config = SyncConfig(remove_silent_entries=True)
        segments_source = item.speech_timeline if item.speech_timeline else None
        cleaned, stats = cleanup_subtitles(
            subtitles,
            item.source_path,
            config,
            precomputed_segments=segments_source,
        )
        if stats.get("status") != "success":
            _update_item(
                batch,
                item,
                message=f"환각 제거 건너뜀 ({stats.get('status')})",
            )
            item.silent_entries_removed = 0
            return False
        cleaned_content = export_srt(cleaned)
        with open(item.transcript_path, "w", encoding="utf-8") as fp:
            fp.write(cleaned_content)
        removed_silent = stats.get("silent_entries_removed", 0)
        item.silent_entries_removed = removed_silent
        summary_msg = f"무음 환각 자막 제거를 완료했습니다. (삭제 {removed_silent}개)"
        _update_item(batch, item, message=summary_msg)
        return True
    except Exception as exc:
        _update_item(
            batch,
            item,
            message=f"무음 환각 제거 실패: {exc}",
            error=str(exc),
        )
        item.silent_entries_removed = 0
        return False
    finally:
        item.download_ready = bool(item.transcript_path and os.path.isfile(item.transcript_path))


def _process_item(batch: WhisperBatch, item: WhisperBatchItem) -> None:
    try:
        storage_key = f"{DEFAULT_STORAGE_KEY}:whisper:{batch.batch_id}:{item.item_id}"
        _update_item(batch, item, status="running", message="세그먼트를 준비하는 중입니다...", progress=0.0)
        source_path = _ensure_source_path(batch, item)
        minutes_per_segment = max(batch.chunk_seconds / 60.0, 1 / 60.0)
        segments = split_video_by_minutes(
            input_path=source_path,
            output_dir=item.segments_dir,
            minutes_per_segment=minutes_per_segment,
            storage_key=storage_key,
            prefix="segment",
        )
        aggregated_segments: List[Dict[str, float]] = []
        for meta in segments:
            for speech in meta.speech_segments or []:
                start = float(speech.get("start", meta.start_time))
                end = float(speech.get("end", meta.end_time))
                if end <= start:
                    continue
                aggregated_segments.append({"start": start, "end": end})
        aggregated_segments.sort(key=lambda seg: seg["start"])
        item.speech_timeline = aggregated_segments
        transcript_path = _transcribe_segments(batch, item, segments)
        item.transcript_path = transcript_path
        item.download_ready = bool(transcript_path and os.path.isfile(transcript_path))

        apply_cleanup = batch.apply_vad_cleanup and item.apply_vad_cleanup
        apply_sync = batch.apply_vad_sync and item.apply_vad_sync

        if apply_sync:
            sync_ok = _apply_vad_sync(batch, item, apply_cleanup=apply_cleanup)
            if sync_ok:
                final_message = "전사 및 싱크 보정이 완료되었습니다."
            else:
                final_message = "전사는 완료했지만 VAD 싱크 보정 결과를 적용하지 못했습니다."
        elif apply_cleanup:
            cleanup_ok = _apply_vad_cleanup_only(batch, item)
            if cleanup_ok:
                final_message = "전사 및 무음 환각 자막 제거를 완료했습니다."
            else:
                final_message = "전사는 완료했지만 무음 환각 자막 제거 결과를 적용하지 못했습니다."
        else:
            item.silent_entries_removed = 0
            final_message = "전사를 완료했습니다. (VAD 후처리 사용 안 함)"
            _update_item(batch, item, message=final_message)

        _update_item(batch, item, status="completed", progress=1.0, message=final_message)
    except Exception as exc:
        _update_item(batch, item, status="failed", message=str(exc), error=str(exc))


def _run_batch(batch: WhisperBatch) -> None:
    _increment_active_batches()
    try:
        for item in batch.items:
            if item.status == "completed":
                continue
            _process_item(batch, item)
        batch.status = "completed"
    except Exception:
        batch.status = "failed"
        raise
    finally:
        batch.touch()
        _decrement_active_batches()


def create_batch(
    *,
    files: Iterable[Optional[FileStorage]],
    chunk_seconds: float = 30.0,
    youtube_urls: Iterable[str] = (),
    apply_vad_cleanup: bool = True,
    apply_vad_sync: bool = True,
) -> Dict[str, Any]:
    uploads = _filter_uploads(files)
    raw_youtube_urls = list(youtube_urls or [])
    normalized_youtube_urls = _normalize_youtube_urls(raw_youtube_urls)
    if not uploads and not normalized_youtube_urls:
        if raw_youtube_urls:
            raise ValueError("유효한 YouTube 링크를 입력해 주세요.")
        raise ValueError("업로드할 영상 또는 오디오 파일을 선택해 주세요.")
    if chunk_seconds <= 0:
        raise ValueError("청크 길이는 0보다 커야 합니다.")

    batch_id = uuid.uuid4().hex
    batch_dir = _ensure_dir(os.path.join(WHISPER_BATCH_ROOT, batch_id))
    uploads_dir = _ensure_dir(os.path.join(batch_dir, "uploads"))
    outputs_dir = _ensure_dir(os.path.join(batch_dir, "outputs"))
    segments_root = _ensure_dir(os.path.join(batch_dir, "segments"))
    youtube_dir = _ensure_dir(os.path.join(batch_dir, "youtube"))

    items: List[WhisperBatchItem] = []
    for uploaded in uploads:
        item_id = uuid.uuid4().hex
        source_path = _save_uploaded_file(uploaded, uploads_dir)
        item_segments_dir = _ensure_dir(os.path.join(segments_root, item_id))
        item_output_dir = _ensure_dir(os.path.join(outputs_dir, item_id))
        display_name = uploaded.filename or os.path.basename(source_path)
        items.append(
            WhisperBatchItem(
                item_id=item_id,
                file_name=display_name,
                source_path=source_path,
                segments_dir=item_segments_dir,
                output_dir=item_output_dir,
                downloads_dir=uploads_dir,
                apply_vad_cleanup=apply_vad_cleanup,
                apply_vad_sync=apply_vad_sync,
            )
        )

    for url in normalized_youtube_urls:
        item_id = uuid.uuid4().hex
        item_segments_dir = _ensure_dir(os.path.join(segments_root, item_id))
        item_output_dir = _ensure_dir(os.path.join(outputs_dir, item_id))
        display_name = _youtube_display_name(url, f"YouTube - {item_id[:6]}")
        items.append(
            WhisperBatchItem(
                item_id=item_id,
                file_name=display_name,
                source_path="",
                segments_dir=item_segments_dir,
                output_dir=item_output_dir,
                youtube_url=url,
                downloads_dir=youtube_dir,
                apply_vad_cleanup=apply_vad_cleanup,
                apply_vad_sync=apply_vad_sync,
            )
        )

    batch = WhisperBatch(
        batch_id=batch_id,
        chunk_seconds=chunk_seconds,
        base_dir=batch_dir,
        apply_vad_cleanup=apply_vad_cleanup,
        apply_vad_sync=apply_vad_sync,
        items=items,
    )
    with _LOCK:
        _BATCHES[batch_id] = batch
    batch.start_worker()
    return batch.to_dict()


def get_batch_state(batch_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        batch = _BATCHES.get(batch_id)
        if not batch:
            return None
        batch.touch()
        return batch.to_dict()


def has_batch(batch_id: str) -> bool:
    with _LOCK:
        return batch_id in _BATCHES


def get_item_transcript_path(batch_id: str, item_id: str) -> Optional[str]:
    with _LOCK:
        batch = _BATCHES.get(batch_id)
        if not batch:
            return None
        for item in batch.items:
            if item.item_id == item_id and item.transcript_path and os.path.isfile(item.transcript_path):
                return item.transcript_path
    return None
