"""Background subtitle generation pipeline built on top of video_split."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from yt_dlp import YoutubeDL

from constants import BASE_DIR, DEFAULT_MODEL
from google.genai import errors as genai_errors
from module import ffmpeg_module, srt_module
from module import storage as storage_module
from module.gemini_module import GeminiClient
from module.video_split import SegmentMetadata, split_video_by_minutes, DEFAULT_STORAGE_KEY
from module.Whisper_util import transcribe_audio_with_timestamps

if TYPE_CHECKING:  # pragma: no cover
    from werkzeug.datastructures import FileStorage


logger = logging.getLogger(__name__)

SUBTITLE_JOB_ROOT = os.path.join(BASE_DIR, "generated_subtitles")
os.makedirs(SUBTITLE_JOB_ROOT, exist_ok=True)
_HISTORY_STORAGE_PREFIX = "subtitle_job_history:"
_QUOTA_COOLDOWN_SECONDS = 10.0

ENTRY_UPDATE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
            "default": [],
        }
    },
    "required": ["entries"],
}


def _now() -> float:
    return time.time()


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0.0, float(seconds)) * 1000
    total_ms = round(total_ms)
    ms = int(total_ms % 1000)
    total_seconds = int(total_ms // 1000)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _truncate_text(value: str, limit: int = 240) -> str:
    text = (value or "").strip()
    if not text or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


_INVALID_FILENAME_CHARS = set('<>:"/\\|?*')


def _sanitize_filename_preserve_unicode(original: str, fallback: str) -> str:
    """업로드된 파일명에서 위험 문자를 제거하면서 한글 등 유니코드 문자를 보존합니다."""
    name = os.path.basename(original or "").strip()
    if not name:
        return fallback

    name = unicodedata.normalize("NFC", name)
    sanitized_chars = []
    for ch in name:
        if ch in _INVALID_FILENAME_CHARS or ord(ch) < 32:
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(ch)
    sanitized = "".join(sanitized_chars).strip()
    sanitized = sanitized.lstrip(".")
    if not sanitized:
        return fallback
    # 윈도우 파일 시스템 최대 길이 고려
    return sanitized[:255]


_RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*'?(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)


def _parse_retry_delay_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str):
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*$", value, re.IGNORECASE)
        if match:
            return max(0.0, float(match.group(1)))
    return None


def _extract_retry_delay_from_payload(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        if "retryDelay" in payload:
            delay = _parse_retry_delay_value(payload.get("retryDelay"))
            if delay is not None:
                return delay
        details = payload.get("details")
        if isinstance(details, list):
            for detail in details:
                delay = _extract_retry_delay_from_payload(detail)
                if delay is not None:
                    return delay
    elif isinstance(payload, list):
        for item in payload:
            delay = _extract_retry_delay_from_payload(item)
            if delay is not None:
                return delay
    return None


def _get_retry_delay_seconds_from_error(error: Exception) -> Optional[float]:
    for attr in ("error", "response", "details"):
        payload = getattr(error, attr, None)
        delay = _extract_retry_delay_from_payload(payload)
        if delay is not None:
            return delay
    text = getattr(error, "message", None) or str(error)
    match = _RETRY_DELAY_PATTERN.search(text)
    if match:
        return max(0.0, float(match.group("seconds")))
    return None


def _send_gemini_with_retry(
    client: GeminiClient,
    *,
    job: Optional["SubtitleJob"],
    context: str,
    send_kwargs: Dict[str, Any],
    max_attempts: int = 3,
) -> Any:
    last_error: Optional[Exception] = None
    non_resource_retry_done = False
    for attempt in range(1, max_attempts + 1):
        try:
            return client.send_message(**send_kwargs)
        except genai_errors.ClientError as exc:
            last_error = exc
            error_text = str(exc)
            if "RESOURCE_EXHAUSTED" in error_text.upper():
                retry_delay = _get_retry_delay_seconds_from_error(exc)
                wait_seconds = max(5.0, (retry_delay or 0.0) + 5.0)
                warning_text = (
                    f"{context} 중 Gemini 쿼터 제한에 도달했습니다. "
                    f"{wait_seconds:.1f}초 후 재시도합니다. (시도 {attempt}/{max_attempts})"
                )
                if job:
                    job.append_log(warning_text, level="warning")
                else:
                    logger.warning(warning_text)
                time.sleep(wait_seconds)
                continue

            if not non_resource_retry_done:
                non_resource_retry_done = True
                retry_delay = _get_retry_delay_seconds_from_error(exc)
                wait_seconds = max(5.0, (retry_delay or 0.0) + 5.0)
                warning_text = (
                    f"{context} 중 오류가 발생했습니다: {error_text}. "
                    f"{wait_seconds:.1f}초 후 재시도합니다. (추가 재시도 1/1)"
                )
                if job:
                    job.append_log(warning_text, level="warning")
                else:
                    logger.warning(warning_text)
                time.sleep(wait_seconds)
                continue
            raise
        except Exception as exc:
            last_error = exc
            if not non_resource_retry_done:
                non_resource_retry_done = True
                wait_seconds = 60.0
                warning_text = (
                    f"{context} 중 알 수 없는 오류가 발생했습니다: {exc}. "
                    f"{wait_seconds:.1f}초 후 재시도합니다. (추가 재시도 1/1)"
                )
                if job:
                    job.append_log(warning_text, level="warning")
                else:
                    logger.warning(warning_text)
                time.sleep(wait_seconds)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Gemini 요청 재시도에 실패했습니다.")


def _history_storage_key(job_id: str) -> str:
    return f"{_HISTORY_STORAGE_PREFIX}{job_id}"


def _reset_saved_history(job_id: str) -> None:
    storage_module.remove_value(_history_storage_key(job_id), None)


def _persist_model_history(job_id: str, client: GeminiClient) -> List[Dict[str, Any]]:
    """클라이언트 히스토리에서 모델 응답만 추려 저장합니다."""
    model_messages: List[Dict[str, Any]] = []
    try:
        for message in client.get_history():
            if message.get("role") != "model":
                continue
            model_messages.append(copy.deepcopy(message))
    except Exception:
        model_messages = []
    storage_module.set_value(_history_storage_key(job_id), model_messages)
    # start_chat 호출 시 외부 변형을 막기 위한 복사본 반환
    return [copy.deepcopy(item) for item in model_messages]


def _load_saved_model_history(job_id: str) -> List[Dict[str, Any]]:
    stored = storage_module.get_value(_history_storage_key(job_id), [])
    if not stored:
        return []
    return [copy.deepcopy(item) for item in stored]


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


def _build_speech_windows(
    segment: SegmentState,
    max_chunk_seconds: float = 10.0,
    pad_before: float = 0.25,
    pad_after: float = 0.25,
) -> List[Dict[str, float]]:
    if not segment.speech_segments:
        return []
    windows: List[Dict[str, float]] = []
    clip_start = segment.start_time
    clip_duration = segment.duration
    for speech in segment.speech_segments:
        try:
            abs_start = max(0.0, float(speech.get("start", 0.0)))
            abs_end = max(abs_start + 0.05, float(speech.get("end", abs_start + 0.1)))
        except (TypeError, ValueError):
            continue
        rel_start = max(0.0, abs_start - clip_start)
        rel_end = max(rel_start + 0.05, abs_end - clip_start)
        current = rel_start
        is_first_chunk = True
        while current < rel_end - 1e-6:
            chunk_start = current
            chunk_end = min(rel_end, current + max_chunk_seconds)
            start_val = chunk_start
            end_val = chunk_end
            if is_first_chunk:
                start_val = max(0.0, chunk_start - pad_before)
            if chunk_end >= rel_end - 1e-6:
                end_val = min(clip_duration, chunk_end + pad_after)
            windows.append({"start": round(start_val, 3), "end": round(end_val, 3)})
            current = chunk_end
            is_first_chunk = False
    return windows


def _build_whisper_windows(segment: SegmentState, max_text_length: int = 80) -> List[Dict[str, Any]]:
    if not segment.whisper_segments:
        return []
    windows: List[Dict[str, Any]] = []
    for entry in segment.whisper_segments:
        try:
            start = max(0.0, float(entry.get("start", 0.0)))
            end = max(start + 0.05, float(entry.get("end", start + 0.1)))
        except (TypeError, ValueError):
            continue
        snippet = _truncate_text(entry.get("text", ""), max_text_length) or ""
        windows.append(
            {
                "index": int(entry.get("index", len(windows) + 1)),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": snippet,
            }
        )
    return windows


def _build_whisper_context(segment: SegmentState, limit: int = 20) -> str:
    if not segment.whisper_segments:
        return ""
    lines = ["Preliminary Whisper transcript (context only):"]
    count = 0
    for entry in segment.whisper_segments:
        if count >= limit:
            break
        text = _truncate_text(entry.get("text", ""), 200)
        if not text:
            continue
        start_ts = entry.get("start_timecode") or _format_timestamp(entry.get("start", 0.0))
        end_ts = entry.get("end_timecode") or _format_timestamp(entry.get("end", 0.0))
        lines.append(f"- [{start_ts} --> {end_ts}] {text}")
        count += 1
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _generate_whisper_segments(segment: SegmentState, language_hint: Optional[str] = None) -> List[Dict[str, Any]]:
    raw_segments = transcribe_audio_with_timestamps(
        segment.file_path,
        language=language_hint,
        show_progress=False,
    )
    formatted: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_segments, start=1):
        try:
            start = max(0.0, float(entry.get("start", 0.0)))
            end = max(start + 0.05, float(entry.get("end", start + 0.1)))
        except (TypeError, ValueError):
            continue
        text = str(entry.get("text", "")).strip()
        clip_index = int(entry.get("index", idx))
        formatted.append(
            {
                "index": clip_index,
                "start": start,
                "end": end,
                "text": text,
                "start_timecode": entry.get("start_timecode") or _format_timestamp(start),
                "end_timecode": entry.get("end_timecode") or _format_timestamp(end),
                "absolute_start": segment.start_time + start,
                "absolute_end": segment.start_time + end,
                "absolute_start_timecode": _format_timestamp(segment.start_time + start),
                "absolute_end_timecode": _format_timestamp(segment.start_time + end),
            }
        )
    return formatted


def _srt_timestamp_to_seconds(timestamp: str) -> float:
    """SRT 타임스탬프(HH:MM:SS,mmm)를 초 단위로 변환합니다."""
    # 00:00:01,500 -> 1.5
    try:
        time_part, ms_part = timestamp.split(',')
        h, m, s = time_part.split(':')
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms_part) / 1000.0
        return total_seconds
    except (ValueError, AttributeError):
        return 0.0


def _create_segments_from_srt(job: "SubtitleJob", srt_path: str) -> List[SegmentMetadata]:
    """업로드된 SRT 파일의 타임스탬프를 기준으로 세그먼트를 생성합니다.

    전체 영상을 chunk_minutes 단위로 시간 구간으로 나누고,
    각 구간에 해당하는 자막 엔트리들을 포함시킵니다.
    """
    # SRT 파일 파싱
    subtitles = srt_module.read_srt(srt_path)
    if not subtitles:
        raise ValueError("SRT 파일에 자막이 없습니다.")

    # 전체 영상의 시작과 끝 시간 파악
    first_start = _srt_timestamp_to_seconds(subtitles[0]['start'])
    last_end = _srt_timestamp_to_seconds(subtitles[-1]['end'])

    job.append_log(f"SRT 파일 범위: {first_start:.2f}s ~ {last_end:.2f}s")

    # chunk_minutes 단위로 시간 구간 나누기
    max_duration = job.chunk_minutes * 60.0
    groups: List[List[Dict[str, float]]] = []

    current_time = first_start
    while current_time < last_end:
        segment_end = min(current_time + max_duration, last_end)

        # 이 시간 구간에 속하는 자막들 찾기
        group = []
        for subtitle in subtitles:
            start_sec = _srt_timestamp_to_seconds(subtitle['start'])
            end_sec = _srt_timestamp_to_seconds(subtitle['end'])

            # 자막이 현재 구간과 겹치는지 확인
            if start_sec < segment_end and end_sec > current_time:
                group.append({
                    'start': start_sec,
                    'end': end_sec,
                })

        if group:
            groups.append(group)

        current_time = segment_end

    # 각 그룹에 대해 비디오 분할 및 SegmentMetadata 생성
    segments_metadata: List[SegmentMetadata] = []
    for idx, group in enumerate(groups):
        start_time = group[0]['start']
        end_time = group[-1]['end']
        duration = end_time - start_time

        # FFmpeg로 비디오 분할
        segment_filename = f"segment_{idx:04d}.mp4"
        segment_path = os.path.join(job.chunks_dir, segment_filename)

        # FFmpeg 명령어 실행 (PATH에 이미 등록되어 있음)
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', job.source_path,
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            segment_path
        ]
        import subprocess
        # Windows 기본 로캘(cp949)에서 FFmpeg 출력 디코딩 오류를 막기 위해 바이너리 모드로 실행
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
            logger.error(f"FFmpeg 분할 실패: {stderr_text}")
            raise RuntimeError(f"세그먼트 {idx} 분할 실패")

        # SegmentMetadata 생성
        metadata = SegmentMetadata(
            index=idx,
            file_path=segment_path,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            speech_segments=[{'start': seg['start'] - start_time, 'end': seg['end'] - start_time} for seg in group]
        )
        segments_metadata.append(metadata)

        job.append_log(f"세그먼트 {idx}: {start_time:.2f}s ~ {end_time:.2f}s ({len(group)}개 자막)")

    return segments_metadata


def _load_entries_from_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    subtitles = srt_module.read_srt(srt_path)
    entries: List[Dict[str, Any]] = []
    for idx, subtitle in enumerate(subtitles, start=1):
        start_sec = _srt_timestamp_to_seconds(subtitle["start"])
        end_sec = _srt_timestamp_to_seconds(subtitle["end"])
        entries.append(
            {
                "index": idx,
                "start": start_sec,
                "end": end_sec,
                "text": subtitle["text"].strip(),
                "original_text": subtitle["text"].strip(),
            }
        )
    return entries


def _build_entries_from_whisper(job: SubtitleJob) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    next_index = 1
    for segment in job.segments:
        try:
            segment.whisper_segments = _generate_whisper_segments(segment)
            if segment.whisper_segments:
                job.append_log(
                    f"{segment.index}번 세그먼트에서 Whisper 엔트리 {len(segment.whisper_segments)}개 확보"
                )
        except Exception as exc:
            segment.whisper_segments = []
            job.append_log(f"세그먼트 {segment.index} Whisper 분석 실패: {exc}", level="warning")
            continue
        for whisper in segment.whisper_segments:
            start = float(whisper.get("absolute_start", segment.start_time + whisper.get("start", 0.0)))
            end = float(whisper.get("absolute_end", segment.start_time + whisper.get("end", 0.0)))
            text = str(whisper.get("text", "")).strip()
            entries.append(
                {
                    "index": next_index,
                    "start": start,
                    "end": end,
                    "text": text,
                    "original_text": text,
                }
            )
            next_index += 1
    return entries


def _initialize_job_entries(job: SubtitleJob, entries: List[Dict[str, Any]]) -> None:
    ordered = sorted(entries, key=lambda item: item["start"])
    for idx, entry in enumerate(ordered, start=1):
        entry["index"] = idx
        entry.setdefault("original_text", entry.get("text", ""))
    job.transcript_entries = ordered
    job.entry_index_map = {entry["index"]: entry for entry in ordered}


def _entries_for_segment(job: SubtitleJob, segment: SegmentState) -> List[Dict[str, Any]]:
    if not job.transcript_entries:
        return []
    selected: List[Dict[str, Any]] = []
    for entry in job.transcript_entries:
        if entry["start"] < segment.end_time and entry["end"] > segment.start_time:
            selected.append(entry)
        elif entry["start"] >= segment.end_time:
            break
    return selected


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


@dataclass
class SegmentState:
    index: int
    start_time: float
    end_time: float
    duration: float
    file_path: str
    speech_segments: List[Dict[str, float]] = field(default_factory=list)
    whisper_segments: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    message: str = ""
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "speech_segments": self.speech_segments,
            "whisper_segments": self.whisper_segments,
            "status": self.status,
            "message": self.message,
            "file_name": os.path.basename(self.file_path),
            "download_available": os.path.isfile(self.file_path),
        }


@dataclass
class SubtitleJob:
    job_id: str
    output_dir: str
    chunks_dir: str
    status: str = "pending"
    phase: str = "initializing"
    message: str = ""
    processed_segments: int = 0
    total_segments: int = 0
    source_path: Optional[str] = None
    youtube_url: Optional[str] = None
    chunk_minutes: float = 10.0
    mode: str = "transcribe"
    target_language: Optional[str] = None
    custom_prompt: Optional[str] = None
    model_name: str = DEFAULT_MODEL
    transcript_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    segments: List[SegmentState] = field(default_factory=list)
    transcript_entries: List[Dict[str, Any]] = field(default_factory=list)
    entry_index_map: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    failed_segments: List[int] = field(default_factory=list)
    stop_requested: bool = False
    voice_override_path: Optional[str] = None

    def progress(self) -> float:
        if self.total_segments <= 0:
            return 0.0
        return min(1.0, max(0.0, self.processed_segments / self.total_segments))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "processed_segments": self.processed_segments,
            "total_segments": self.total_segments,
            "progress": self.progress(),
            "source_file": os.path.basename(self.source_path) if self.source_path else None,
            "youtube_url": self.youtube_url,
            "chunk_minutes": self.chunk_minutes,
            "mode": self.mode,
            "target_language": self.target_language,
            "custom_prompt": self.custom_prompt,
            "model_name": self.model_name,
            "transcript_ready": bool(self.transcript_path and os.path.isfile(self.transcript_path)),
            "transcript_path": self.transcript_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "segments": [segment.to_dict() for segment in self.segments],
            "logs": list(self.logs),
            "failed_segments": list(self.failed_segments),
            "stop_requested": self.stop_requested,
        }

    def append_log(self, text: str, level: str = "info") -> None:
        entry = {
            "timestamp": _now(),
            "level": level,
            "message": text,
        }
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs[:] = self.logs[-200:]


_JOBS: Dict[str, SubtitleJob] = {}
_LOCK = threading.RLock()


def _get_job(job_id: str) -> Optional[SubtitleJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def get_job_data(job_id: str) -> Optional[Dict[str, Any]]:
    job = _get_job(job_id)
    if not job:
        return None
    return job.to_dict()


def get_transcript_path(job_id: str) -> Optional[str]:
    job = _get_job(job_id)
    if not job or not job.transcript_path:
        return None
    if os.path.isfile(job.transcript_path):
        return job.transcript_path
    return None


def get_segment_path(job_id: str, segment_index: int) -> Optional[str]:
    job = _get_job(job_id)
    if not job:
        return None
    for segment in job.segments:
        if segment.index == segment_index:
            return segment.file_path if os.path.isfile(segment.file_path) else None
    return None


def request_stop(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return False
        if job.status in {"completed", "failed", "cancelled"}:
            return False
        already_requested = job.stop_requested
        job.stop_requested = True

    if not already_requested:
        job.append_log("사용자가 작업 중지를 요청했습니다.", level="warning")
        _update_job(job, phase="stopping", message="작업 중지 요청을 처리하고 있습니다.")
    return True


def _update_job(job: SubtitleJob, *, status: Optional[str] = None, phase: Optional[str] = None,
                message: Optional[str] = None, error: Optional[str] = None) -> None:
    if status:
        job.status = status
    if phase:
        job.phase = phase
    if message is not None:
        job.message = message
        job.append_log(message)
    if error is not None:
        job.error = error
        if error:
            job.append_log(error, level="error")
    job.updated_at = _now()


def _save_uploaded_file(uploaded_file: "FileStorage", dest_dir: str) -> str:
    original_name = uploaded_file.filename or ""
    _, ext = os.path.splitext(original_name)
    default_ext = ext if ext else ".bin"
    fallback_name = f"upload_{uuid.uuid4().hex}{default_ext}"
    filename = _sanitize_filename_preserve_unicode(original_name, fallback_name)
    _ensure_dir(dest_dir)
    dest = os.path.join(dest_dir, filename)
    uploaded_file.save(dest)
    return dest


def _prepare_source_path(job: SubtitleJob, youtube_url: Optional[str],
                         uploaded_path: Optional[str]) -> str:
    if uploaded_path:
        job.append_log("업로드된 영상 파일을 사용합니다.")
        return uploaded_path
    if youtube_url:
        job.append_log("YouTube 영상을 다운로드하는 중입니다.")
        return _download_youtube_video(youtube_url, os.path.join(job.output_dir, "download"))
    raise ValueError("영상 파일 또는 YouTube 링크가 필요합니다.")


def _merge_video_with_voice_track(video_path: str, audio_path: str, output_dir: str) -> str:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

    ffmpeg_module.register_ffmpeg_path()
    _ensure_dir(output_dir)
    merged_path = os.path.join(output_dir, f"voice_overlay_{uuid.uuid4().hex}.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        merged_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"음성 교체 작업 실패: {stderr.strip() or exc}") from exc
    return merged_path


def _create_segments(job: SubtitleJob, minutes_per_segment: float) -> List[SegmentMetadata]:
    storage_key = f"{DEFAULT_STORAGE_KEY}:{job.job_id}"
    return split_video_by_minutes(
        input_path=job.source_path or "",
        output_dir=job.chunks_dir,
        minutes_per_segment=minutes_per_segment,
        storage_key=storage_key,
    )


def start_job(
    *,
    youtube_url: Optional[str],
    uploaded_file: Optional["FileStorage"],
    voice_file: Optional["FileStorage"],
    srt_file: Optional["FileStorage"],
    chunk_minutes: float,
    mode: str,
    target_language: Optional[str],
    custom_prompt: Optional[str],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if chunk_minutes <= 0:
        raise ValueError("청크 길이는 0보다 커야 합니다.")
    if mode not in {"transcribe", "translate"}:
        raise ValueError("지원하지 않는 전사 모드입니다.")
    if mode == "translate" and not target_language:
        raise ValueError("번역 모드에서는 번역 대상 언어가 필요합니다.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Google API Key가 설정되지 않았습니다.")

    if not (youtube_url or (uploaded_file and uploaded_file.filename)):
        raise ValueError("영상 파일 또는 YouTube 링크 중 하나가 필요합니다.")

    job_id = uuid.uuid4().hex
    output_dir = _ensure_dir(os.path.join(SUBTITLE_JOB_ROOT, job_id))
    chunks_dir = _ensure_dir(os.path.join(output_dir, "segments"))
    uploaded_path: Optional[str] = None
    srt_path: Optional[str] = None
    voice_path: Optional[str] = None
    if uploaded_file and uploaded_file.filename:
        uploaded_path = _save_uploaded_file(uploaded_file, os.path.join(output_dir, "source"))
    if srt_file and srt_file.filename:
        srt_path = _save_uploaded_file(srt_file, os.path.join(output_dir, "source"))
    if voice_file and voice_file.filename:
        voice_path = _save_uploaded_file(voice_file, os.path.join(output_dir, "source"))
    job = SubtitleJob(
        job_id=job_id,
        output_dir=output_dir,
        chunks_dir=chunks_dir,
        chunk_minutes=chunk_minutes,
        mode=mode,
        target_language=target_language,
        youtube_url=youtube_url,
        custom_prompt=custom_prompt,
        model_name=model_name,
        voice_override_path=voice_path,
    )
    _reset_saved_history(job_id)

    with _LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        args=(job, youtube_url, uploaded_path, srt_path, voice_path),
        daemon=True,
    )
    thread.start()

    return job.to_dict()


def _process_segment_entries(
    job: SubtitleJob,
    client: GeminiClient,
    segment: SegmentState,
    segment_entries: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    if not segment_entries:
        return 0, []
    task_label = "translate" if job.mode == "translate" else "transcribe"
    instruction_lines = [
        "You will receive a video segment and the fixed subtitle entries that belong to the same time range.",
        "Each entry already has the correct start/end timestamps. Update only the text field while keeping the given indexes.",
    ]
    if job.mode == "translate" and job.target_language:
        instruction_lines.append(
            f"Translate every entry into {job.target_language} while keeping the speaker's intent and natural tone."
        )
        instruction_lines.append("Return only the translated sentences without source text or language labels.")
        instruction_lines.append("Use the timestamps below as context, but only update the text for each entry.")
    else:
        instruction_lines.append("Provide a faithful transcription in the original spoken language.")
    instruction_lines.append(
        "Respond with JSON that follows the schema: { \"entries\": [ { \"index\": number, \"text\": string } ] }."
    )
    if job.custom_prompt:
        instruction_lines.append(f"Additional instructions: {job.custom_prompt}")
    instruction_lines.append("\nEntries:")
    for entry in segment_entries:
        start_ts = _format_timestamp(entry["start"])
        end_ts = _format_timestamp(entry["end"])
        preview = _truncate_text(entry.get("text", ""), 200)
        instruction_lines.append(f"- #{entry['index']} [{start_ts} --> {end_ts}]: {preview}")
    prompt = "\n".join(instruction_lines)

    response = _send_gemini_with_retry(
        client,
        job=job,
        context=f"{segment.index}번 세그먼트 {task_label}",
        send_kwargs={
            "message": prompt,
            "media_paths": [segment.file_path],
            "response_schema": ENTRY_UPDATE_RESPONSE_SCHEMA,
            "max_wait_seconds": 900,
        },
    )
    _persist_model_history(job.job_id, client)
    if isinstance(response, str):
        try:
            payload = json.loads(response or "{}")
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {}
    updates = payload.get("entries", []) or []
    updated_records: List[Dict[str, Any]] = []
    update_count = 0
    for update in updates:
        try:
            entry_index = int(update.get("index"))
        except (TypeError, ValueError):
            continue
        text = str(update.get("text", "")).strip()
        if not text:
            continue
        entry_ref = job.entry_index_map.get(entry_index)
        if not entry_ref:
            continue
        entry_ref["text"] = text
        updated_records.append(
            {
                "index": entry_index,
                "start": entry_ref["start"],
                "end": entry_ref["end"],
                "text": text,
            }
        )
        update_count += 1
    return update_count, updated_records


def _process_segment(
    job: SubtitleJob,
    client: GeminiClient,
    segment: SegmentState,
    *,
    task_label: str,
    last_segment_start_at: Optional[float],
    allow_processed_increment: bool,
    is_retry: bool = False,
) -> float:
    if last_segment_start_at is not None:
        elapsed = max(0.0, time.time() - last_segment_start_at)
        if elapsed < _QUOTA_COOLDOWN_SECONDS:
            wait_seconds = _QUOTA_COOLDOWN_SECONDS - elapsed
            job.append_log(f"쿼터 보호를 위해 {wait_seconds:.1f}초 대기 후 다음 세그먼트를 처리합니다.")
            time.sleep(wait_seconds)

    started_at = time.time()
    segment.status = "processing"
    if is_retry:
        segment.message = f"재시도 {task_label} 중..."
        job.append_log(f"{segment.index}번 세그먼트 재시도 {task_label} 중...")
    else:
        segment.message = f"{task_label} 중..."
        job.append_log(f"{segment.index}번 세그먼트 {task_label} 중...")
    job.updated_at = _now()

    segment_entries = _entries_for_segment(job, segment)
    if not segment_entries:
        segment.status = "completed"
        segment.message = "처리할 자막 없음"
        if segment.index in job.failed_segments:
            job.failed_segments.remove(segment.index)
        if allow_processed_increment:
            job.processed_segments += 1
        job.updated_at = _now()
        return started_at

    try:
        updated_count, updated_records = _process_segment_entries(job, client, segment, segment_entries)
        if updated_count == 0:
            segment.message = "재시도 응답 없음 (원본 유지)" if is_retry else "응답 없음 (원본 유지)"
        else:
            base_message = f"{updated_count}개 엔트리 {task_label} 완료"
            if is_retry:
                base_message = f"재시도 성공: {base_message}"
            segment.message = base_message
        segment.transcript = updated_records or [
            {
                "index": entry["index"],
                "start": entry["start"],
                "end": entry["end"],
                "text": entry.get("text", ""),
            }
            for entry in segment_entries
        ]
        segment.status = "completed"
        if segment.index in job.failed_segments:
            job.failed_segments.remove(segment.index)
        success_log = f"{segment.index}번 세그먼트 {task_label} 완료: {updated_count}개 업데이트"
        if is_retry:
            success_log += " (재시도)"
        job.append_log(success_log)
    except Exception as exc:
        logger.exception("세그먼트 %s 처리 실패", segment.index)
        failure_log = (
            f"{segment.index}번 세그먼트 재시도 처리 실패: {exc}. 원본 텍스트를 유지합니다."
            if is_retry
            else f"{segment.index}번 세그먼트 처리 실패: {exc}. 원본 텍스트를 유지합니다."
        )
        job.append_log(failure_log, level="warning")
        segment.transcript = [
            {
                "index": entry["index"],
                "start": entry["start"],
                "end": entry["end"],
                "text": entry.get("text", ""),
            }
            for entry in segment_entries
        ]
        segment.status = "error"
        segment.message = "재시도 실패 (원본 유지)" if is_retry else "실패 (원본 유지)"
        if segment.index not in job.failed_segments:
            job.failed_segments.append(segment.index)
    finally:
        if allow_processed_increment:
            job.processed_segments += 1
        job.updated_at = _now()
        model_history = _load_saved_model_history(job.job_id)
        client.start_chat(history=model_history, suppress_log=True, reset_usage=False)

    return started_at


def _mark_segments_cancelled(job: SubtitleJob) -> None:
    for segment in job.segments:
        if segment.status not in {"completed", "error", "cancelled"}:
            segment.status = "cancelled"
            segment.message = "사용자 요청으로 중지됨"


def _run_job(
    job: SubtitleJob,
    youtube_url: Optional[str],
    uploaded_path: Optional[str],
    srt_path: Optional[str] = None,
    voice_path: Optional[str] = None,
) -> None:
    try:
        _update_job(job, status="running", phase="source", message="영상 소스를 준비하는 중입니다.")
        job.source_path = _prepare_source_path(job, youtube_url, uploaded_path)
        job.append_log(f"소스 파일: {job.source_path}")

        if voice_path:
            job.append_log("업로드된 음성 파일로 영상 오디오를 교체합니다.")
            try:
                replaced_path = _merge_video_with_voice_track(
                    job.source_path,
                    voice_path,
                    os.path.join(job.output_dir, "source"),
                )
                job.source_path = replaced_path
                job.append_log("음성 교체가 완료되어 새 영상 파일을 사용합니다.")
            except Exception as exc:
                job.append_log(f"음성 교체 실패: {exc}. 원본 오디오로 계속 진행합니다.", level="warning")

        _update_job(job, phase="split", message="영상 분할을 시작합니다.")
        ffmpeg_module.register_ffmpeg_path()

        if srt_path:
            job.append_log(f"SRT 타임스탬프를 기반으로 세그먼트를 생성합니다: {srt_path}")
            segments_metadata = _create_segments_from_srt(job, srt_path)
        else:
            segments_metadata = _create_segments(job, job.chunk_minutes)

        job.total_segments = len(segments_metadata)
        if job.total_segments == 0:
            raise ValueError("생성된 세그먼트가 없습니다.")

        job.segments = [
            SegmentState(
                index=meta.index,
                start_time=meta.start_time,
                end_time=meta.end_time,
                duration=meta.duration,
                file_path=meta.file_path,
                speech_segments=meta.speech_segments,
            )
            for meta in segments_metadata
        ]
        job.append_log(f"{job.total_segments}개의 세그먼트가 생성되었습니다.")

        _update_job(job, phase="entries", message="기본 자막 엔트리를 준비합니다.")
        if srt_path:
            base_entries = _load_entries_from_srt_file(srt_path)
            job.append_log(f"SRT 자막 엔트리 {len(base_entries)}개를 불러왔습니다.")
        else:
            job.append_log("Whisper를 사용해 기본 자막 엔트리를 생성합니다.")
            base_entries = _build_entries_from_whisper(job)
            job.append_log(f"Whisper 자막 엔트리 {len(base_entries)}개를 확보했습니다.")
        if not base_entries:
            raise ValueError("처리할 자막 엔트리를 생성하지 못했습니다.")
        _initialize_job_entries(job, base_entries)
        job.append_log("자막 엔트리 초기화 완료.")

        _update_job(job, phase="llm", message="LLM이 영상과 엔트리를 함께 분석합니다.")
        client = GeminiClient(model=job.model_name, thinking_budget=-1)
        client.start_chat()
        last_segment_start_at: Optional[float] = None
        task_label = "번역" if job.mode == "translate" else "전사"

        for segment in job.segments:
            if job.stop_requested:
                job.append_log("사용자 요청으로 작업을 중지합니다.", level="warning")
                _mark_segments_cancelled(job)
                _update_job(job, status="cancelled", phase="stopped", message="사용자 요청으로 작업이 중지되었습니다.")
                return
            last_segment_start_at = _process_segment(
                job,
                client,
                segment,
                task_label=task_label,
                last_segment_start_at=last_segment_start_at,
                allow_processed_increment=True,
                is_retry=False,
            )

        if job.stop_requested:
            job.append_log("사용자 요청으로 작업을 중지합니다.", level="warning")
            _mark_segments_cancelled(job)
            _update_job(job, status="cancelled", phase="stopped", message="사용자 요청으로 작업이 중지되었습니다.")
            return

        initial_failures = sorted(job.failed_segments)
        if initial_failures:
            failure_list_text = ", ".join(str(idx) for idx in initial_failures)
            job.append_log(f"실패한 세그먼트 확인: {failure_list_text}", level="warning")
            _update_job(job, phase="retry", message="실패한 세그먼트를 재시도하는 중입니다.")
            for segment_index in initial_failures:
                segment = next((seg for seg in job.segments if seg.index == segment_index), None)
                if not segment:
                    continue
                if job.stop_requested:
                    job.append_log("재시도 중 사용자 요청으로 작업을 중지합니다.", level="warning")
                    _mark_segments_cancelled(job)
                    _update_job(job, status="cancelled", phase="stopped", message="사용자 요청으로 작업이 중지되었습니다.")
                    return
                last_segment_start_at = _process_segment(
                    job,
                    client,
                    segment,
                    task_label=task_label,
                    last_segment_start_at=last_segment_start_at,
                    allow_processed_increment=False,
                    is_retry=True,
                )
            if job.stop_requested:
                job.append_log("재시도 중 사용자 요청으로 작업을 중지합니다.", level="warning")
                _mark_segments_cancelled(job)
                _update_job(job, status="cancelled", phase="stopped", message="사용자 요청으로 작업이 중지되었습니다.")
                return
            if job.failed_segments:
                remaining_text = ", ".join(str(idx) for idx in job.failed_segments)
                job.append_log(f"재시도에도 실패한 세그먼트: {remaining_text}", level="warning")
                job.message = "일부 세그먼트는 재시도에도 실패했습니다."
            else:
                job.append_log("실패한 세그먼트를 재시도하여 성공했습니다.")
                job.message = "실패 세그먼트 재시도가 완료되었습니다."
            job.updated_at = _now()

        combined = sorted(job.transcript_entries, key=lambda item: item["start"])
        if not combined:
            job.append_log("최종 자막 엔트리가 없습니다.")
        output_path = os.path.join(job.output_dir, f"{job.job_id}.srt")
        job.transcript_path = _write_srt(combined, output_path)
        if job.failed_segments:
            final_message = (
                f"{task_label} 작업이 완료되었지만 {len(job.failed_segments)}개 세그먼트는 실패했습니다."
            )
        else:
            final_message = f"{task_label} 작업이 완료되었습니다."
        _update_job(job, status="completed", phase="finished", message=final_message)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Subtitle job %s failed", job.job_id)
        _update_job(job, status="failed", phase="error", message="작업 중 오류가 발생했습니다.", error=str(exc))
    finally:
        _reset_saved_history(job.job_id)


__all__ = [
    "start_job",
    "get_job_data",
    "get_transcript_path",
    "get_segment_path",
    "request_stop",
]
