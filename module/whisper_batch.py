from __future__ import annotations

import os
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from werkzeug.datastructures import FileStorage

from constants import BASE_DIR
from module.video_split import (
    DEFAULT_STORAGE_KEY,
    SegmentMetadata,
    split_video_by_minutes,
)
from module.Whisper_util import WhisperUtil


WHISPER_BATCH_ROOT = os.path.join(BASE_DIR, "whisper_batches")
os.makedirs(WHISPER_BATCH_ROOT, exist_ok=True)

_BATCHES: Dict[str, "WhisperBatch"] = {}
_LOCK = threading.RLock()
_WHISPER_LOCK = threading.Lock()
_WHISPER_INSTANCE: Optional[WhisperUtil] = None


def _get_whisper() -> WhisperUtil:
    global _WHISPER_INSTANCE
    with _WHISPER_LOCK:
        if _WHISPER_INSTANCE is None:
            _WHISPER_INSTANCE = WhisperUtil()
        return _WHISPER_INSTANCE


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_INVALID_CHARS = set('<>:"/\\|?*\0')


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
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    transcript_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

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
            "transcript_ready": bool(download_url),
            "download_url": download_url,
            "updated_at": self.updated_at,
        }


@dataclass
class WhisperBatch:
    batch_id: str
    chunk_minutes: float
    base_dir: str
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
            "chunk_minutes": self.chunk_minutes,
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
        segments = split_video_by_minutes(
            input_path=item.source_path,
            output_dir=item.segments_dir,
            minutes_per_segment=batch.chunk_minutes,
            storage_key=storage_key,
            prefix="segment",
        )
        transcript_path = _transcribe_segments(batch, item, segments)
        item.transcript_path = transcript_path
        _update_item(batch, item, status="completed", progress=1.0, message="전사가 완료되었습니다.")
    except Exception as exc:
        _update_item(batch, item, status="failed", message=str(exc), error=str(exc))


def _run_batch(batch: WhisperBatch) -> None:
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


def create_batch(
    *,
    files: Iterable[Optional[FileStorage]],
    chunk_minutes: float = 10.0,
) -> Dict[str, Any]:
    uploads = _filter_uploads(files)
    if not uploads:
        raise ValueError("업로드할 영상 또는 오디오 파일을 선택해 주세요.")
    if chunk_minutes <= 0:
        raise ValueError("청크 길이는 0보다 커야 합니다.")

    batch_id = uuid.uuid4().hex
    batch_dir = _ensure_dir(os.path.join(WHISPER_BATCH_ROOT, batch_id))
    uploads_dir = _ensure_dir(os.path.join(batch_dir, "uploads"))
    outputs_dir = _ensure_dir(os.path.join(batch_dir, "outputs"))
    segments_root = _ensure_dir(os.path.join(batch_dir, "segments"))

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
            )
        )

    batch = WhisperBatch(
        batch_id=batch_id,
        chunk_minutes=chunk_minutes,
        base_dir=batch_dir,
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
