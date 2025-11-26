from __future__ import annotations

import os
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
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
from module.hallucination_filter import fix_repetitive_hallucinations
from module.ffmpeg_module import get_duration_seconds


WHISPER_BATCH_ROOT = os.path.join(BASE_DIR, "whisper_batches")
os.makedirs(WHISPER_BATCH_ROOT, exist_ok=True)

_BATCHES: Dict[str, "WhisperBatch"] = {}
_LOCK = threading.RLock()
_WHISPER_LOCK = threading.Lock()
_WHISPER_INSTANCE: Optional[WhisperUtil] = None
_ACTIVE_BATCHES = 0
_SUBSCRIBER_LOCK = threading.RLock()
_BATCH_SUBSCRIBERS: Dict[str, List[Queue]] = {}


def _enqueue_state(target_queue: Queue, payload: Dict[str, Any]) -> None:
    try:
        target_queue.put_nowait(payload)
    except Full:
        try:
            target_queue.get_nowait()
        except Empty:
            pass
        try:
            target_queue.put_nowait(payload)
        except Full:
            pass


def _publish_batch_state(batch_id: str, payload: Dict[str, Any]) -> None:
    with _SUBSCRIBER_LOCK:
        subscribers = list(_BATCH_SUBSCRIBERS.get(batch_id, ()))
    for subscriber in subscribers:
        _enqueue_state(subscriber, payload)


def subscribe_to_batch_updates(batch_id: str) -> Queue:
    subscriber: Queue = Queue(maxsize=32)
    with _SUBSCRIBER_LOCK:
        _BATCH_SUBSCRIBERS.setdefault(batch_id, []).append(subscriber)
    return subscriber


def unsubscribe_from_batch_updates(batch_id: str, subscriber: Queue) -> None:
    with _SUBSCRIBER_LOCK:
        listeners = _BATCH_SUBSCRIBERS.get(batch_id)
        if not listeners:
            return
        try:
            listeners.remove(subscriber)
        except ValueError:
            pass
        if not listeners:
            _BATCH_SUBSCRIBERS.pop(batch_id, None)


def _broadcast_batch_state(batch: "WhisperBatch") -> None:
    try:
        payload = batch.to_dict()
    except Exception:
        return
    _publish_batch_state(batch.batch_id, payload)


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
    speech_timeline: List[Dict[str, float]] = field(default_factory=list)
    apply_hallucination_cleanup: bool = True
    segment_progress: List[Dict[str, Any]] = field(default_factory=list)

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
            "apply_hallucination_cleanup": self.apply_hallucination_cleanup,
            "segment_progress": self.segment_progress,
        }


@dataclass
class WhisperBatch:
    batch_id: str
    chunk_seconds: float
    base_dir: str
    apply_hallucination_cleanup: bool = True
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
            "apply_hallucination_cleanup": self.apply_hallucination_cleanup,
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


def _initialize_segment_progress(
    batch: WhisperBatch,
    item: WhisperBatchItem,
    segments: List[SegmentMetadata],
) -> None:
    details: List[Dict[str, Any]] = []
    for idx, meta in enumerate(segments, start=1):
        details.append(
            {
                "segment_index": idx,
                "start_time": float(getattr(meta, "start_time", 0.0) or 0.0),
                "end_time": float(getattr(meta, "end_time", 0.0) or 0.0),
                "start_timecode": _format_timestamp(getattr(meta, "start_time", 0.0) or 0.0),
                "end_timecode": _format_timestamp(getattr(meta, "end_time", 0.0) or 0.0),
                "duration": float(getattr(meta, "duration", 0.0) or 0.0),
                "status": "pending",
                "progress": 0.0,
                "message": "대기 중",
                "updated_at": time.time(),
            }
        )
    item.segment_progress = details
    if details:
        batch.touch()
        _broadcast_batch_state(batch)


def _update_segment_progress(
    batch: WhisperBatch,
    item: WhisperBatchItem,
    segment_idx: int,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    if segment_idx < 0 or segment_idx >= len(item.segment_progress):
        return
    entry = item.segment_progress[segment_idx]
    if status:
        entry["status"] = status
    if progress is not None:
        entry["progress"] = max(0.0, min(1.0, progress))
    if message is not None:
        entry["message"] = message
    entry["updated_at"] = time.time()
    batch.touch()
    _broadcast_batch_state(batch)


def _mark_all_segments_failed(batch: WhisperBatch, item: WhisperBatchItem, error: str) -> None:
    changed = False
    for entry in item.segment_progress:
        if entry.get("status") not in {"completed", "failed"}:
            entry["status"] = "failed"
            entry["progress"] = entry.get("progress", 0.0)
            entry["message"] = error
            entry["updated_at"] = time.time()
            changed = True
    if changed:
        batch.touch()
        _broadcast_batch_state(batch)


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
    _broadcast_batch_state(batch)


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
    *,
    apply_hallucination_cleanup: bool,
) -> str:
    whisper = _get_whisper()
    entries: List[Dict[str, Any]] = []
    total = len(segments) or 1

    def _build_segment_progress_callback(segment_idx: int, duration_seconds: float):
        safe_duration = max(duration_seconds, 0.001)

        def _callback(snippet: Dict[str, Any]) -> None:
            snippet_end = float(snippet.get("end", 0.0) or 0.0)
            ratio = max(0.0, min(1.0, snippet_end / safe_duration))
            preview = str(snippet.get("text", "")).strip() or "전사 중"
            if len(preview) > 80:
                preview = preview[:77] + "..."
            _update_segment_progress(
                batch,
                item,
                segment_idx,
                status="running",
                progress=ratio,
                message=preview,
            )

        return _callback

    for idx, segment in enumerate(segments, start=1):
        _update_item(
            batch,
            item,
            progress=(idx - 1) / total,
            message=f"{idx}/{total}개 세그먼트 전사 중...",
        )
        _update_segment_progress(
            batch,
            item,
            idx - 1,
            status="running",
            progress=0.0,
            message="전사 중",
        )
        segment_duration = float(getattr(segment, "duration", 0.0) or 0.0)
        if segment_duration <= 0.0:
            segment_duration = float(segment.end_time - segment.start_time)
        if segment_duration <= 0.0:
            segment_duration = 0.001
        try:
            progress_callback = _build_segment_progress_callback(idx - 1, segment_duration)
            result = whisper.transcribe_audio(
                segment.file_path,
                show_progress=False,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            _update_segment_progress(
                batch,
                item,
                idx - 1,
                status="failed",
                message=str(exc),
            )
            raise
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
        _update_segment_progress(
            batch,
            item,
            idx - 1,
            status="completed",
            progress=1.0,
            message="완료",
        )
    if not entries:
        entries.append({"start": 0.0, "end": 0.5, "text": "[NO SPEECH DETECTED]"})
    entries.sort(key=lambda entry: entry["start"])
    
    print(f"DEBUG: Transcription finished. Entries: {len(entries)}")

    if apply_hallucination_cleanup:
        print("DEBUG: Starting hallucination cleanup")
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
            print("DEBUG: Hallucination cleanup finished")
        except Exception as exc:
            print(f"DEBUG: Hallucination cleanup failed: {exc}")
            _update_item(batch, item, message=f"환각 반복 검사 건너뜀: {exc}")
    
    print("DEBUG: Writing SRT")
    transcript_dir = _ensure_dir(item.output_dir)
    base_name = os.path.splitext(item.file_name)[0] or item.item_id
    target_name = _sanitize_filename(f"{base_name}.srt", f"{item.item_id}.srt")
    dest = os.path.join(transcript_dir, target_name)
    _write_srt(entries, dest)
    return dest


def _process_item(batch: WhisperBatch, item: WhisperBatchItem) -> None:
    try:
        storage_key = f"{DEFAULT_STORAGE_KEY}:whisper:{batch.batch_id}:{item.item_id}"
        _update_item(batch, item, status="running", message="세그먼트를 준비하는 중입니다...", progress=0.0)
        source_path = _ensure_source_path(batch, item)
        if batch.chunk_seconds <= 0:
            # Bypass splitting; treat the full source file as a single segment
            try:
                video_duration = get_duration_seconds(source_path)
            except Exception:
                video_duration = 0.0
            segments = [
                SegmentMetadata(
                    index=1,
                    file_path=os.path.abspath(source_path),
                    start_time=0.0,
                    end_time=round(float(video_duration or 0.0), 3),
                    duration=round(float(video_duration or 0.0), 3),
                    speech_segments=[],
                )
            ]
        else:
            minutes_per_segment = max(batch.chunk_seconds / 60.0, 1 / 60.0)
            segments = split_video_by_minutes(
                input_path=source_path,
                output_dir=item.segments_dir,
                minutes_per_segment=minutes_per_segment,
                storage_key=storage_key,
                prefix="segment",
            )
        _initialize_segment_progress(batch, item, segments)
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
        apply_hallucination_cleanup = batch.apply_hallucination_cleanup and item.apply_hallucination_cleanup
        transcript_path = _transcribe_segments(
            batch,
            item,
            segments,
            apply_hallucination_cleanup=apply_hallucination_cleanup,
        )
        item.transcript_path = transcript_path
        item.download_ready = bool(transcript_path and os.path.isfile(transcript_path))

        final_message = "전사를 완료했습니다."
        _update_item(batch, item, status="completed", progress=1.0, message=final_message)
    except Exception as exc:
        _mark_all_segments_failed(batch, item, str(exc))
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
        _broadcast_batch_state(batch)
        _decrement_active_batches()


def create_batch(
    *,
    files: Iterable[Optional[FileStorage]],
    chunk_seconds: float = 30.0,
    youtube_urls: Iterable[str] = (),
    apply_hallucination_cleanup: bool = True,
) -> Dict[str, Any]:
    uploads = _filter_uploads(files)
    raw_youtube_urls = list(youtube_urls or [])
    normalized_youtube_urls = _normalize_youtube_urls(raw_youtube_urls)
    if not uploads and not normalized_youtube_urls:
        if raw_youtube_urls:
            raise ValueError("유효한 YouTube 링크를 입력해 주세요.")
        raise ValueError("업로드할 영상 또는 오디오 파일을 선택해 주세요.")
    if chunk_seconds < 0:
        raise ValueError("청크 길이는 음수일 수 없습니다.")

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
                apply_hallucination_cleanup=apply_hallucination_cleanup,
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
                apply_hallucination_cleanup=apply_hallucination_cleanup,
            )
        )

    batch = WhisperBatch(
        batch_id=batch_id,
        chunk_seconds=chunk_seconds,
        base_dir=batch_dir,
        apply_hallucination_cleanup=apply_hallucination_cleanup,
        items=items,
    )
    with _LOCK:
        _BATCHES[batch_id] = batch
    _broadcast_batch_state(batch)
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
