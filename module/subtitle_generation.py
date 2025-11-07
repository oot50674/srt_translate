"""Background subtitle generation pipeline built on top of video_split."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from werkzeug.utils import secure_filename
from yt_dlp import YoutubeDL

from constants import BASE_DIR, DEFAULT_MODEL
from google.genai import errors as genai_errors
from module import ffmpeg_module, srt_module
from module.gemini_module import GeminiClient
from module.video_split import SegmentMetadata, split_video_by_minutes, DEFAULT_STORAGE_KEY
from module.Whisper_util import transcribe_audio_with_timestamps

if TYPE_CHECKING:  # pragma: no cover
    from werkzeug.datastructures import FileStorage


logger = logging.getLogger(__name__)

SUBTITLE_JOB_ROOT = os.path.join(BASE_DIR, "generated_subtitles")
os.makedirs(SUBTITLE_JOB_ROOT, exist_ok=True)

TRANSCRIPT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
            "default": [],
        }
    },
    "required": ["segments"],
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
    for attempt in range(1, max_attempts + 1):
        try:
            return client.send_message(**send_kwargs)
        except genai_errors.ClientError as exc:
            last_error = exc
            error_text = str(exc)
            if "RESOURCE_EXHAUSTED" not in error_text.upper():
                raise
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
    if last_error:
        raise last_error
    raise RuntimeError("Gemini 요청 재시도에 실패했습니다.")


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


def _build_transcript_context(entries: List[Dict[str, Any]], limit: int = 20) -> str:
    if not entries:
        return ""
    recent = entries[-limit:]
    lines = ["Previous transcript context (latest entries):"]
    for item in recent:
        try:
            start = item.get("start", 0.0)
            text = (item.get("text") or "").strip()
            if not text:
                continue
            start_ts = _format_timestamp(start)
            if len(text) > 240:
                text = text[:237].rstrip() + "..."
            lines.append(f"- [{start_ts}] {text}")
        except Exception:
            continue
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg 분할 실패: {result.stderr}")
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
    keep_original_entries: bool = False
    transcript_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    segments: List[SegmentState] = field(default_factory=list)
    transcript_entries: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)

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
            "transcript_ready": bool(self.transcript_path and os.path.isfile(self.transcript_path)),
            "transcript_path": self.transcript_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "segments": [segment.to_dict() for segment in self.segments],
            "logs": list(self.logs),
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
    filename = secure_filename(uploaded_file.filename or "upload.mp4") or f"upload_{uuid.uuid4().hex}.mp4"
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


def _create_segments(job: SubtitleJob, minutes_per_segment: float) -> List[SegmentMetadata]:
    storage_key = f"{DEFAULT_STORAGE_KEY}:{job.job_id}"
    return split_video_by_minutes(
        input_path=job.source_path or "",
        output_dir=job.chunks_dir,
        minutes_per_segment=minutes_per_segment,
        storage_key=storage_key,
    )


def _transcribe_segment(job: SubtitleJob, client: GeminiClient, segment: SegmentState,
                        mode: str, target_language: Optional[str], custom_prompt: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    instruction_lines = [
        "You will receive a short video or audio clip.",
        "Return a JSON object following the provided schema with precise `start` and `end` timestamps in seconds relative to the beginning of this clip.",
        "The `text` must contain a faithful transcription of the spoken content.",
    ]
    if mode == "translate" and target_language:
        instruction_lines.append(
            f"Translate the utterances into {target_language} while keeping speaker intent."
        )
        instruction_lines.append(
            f"The `text` field must contain only the final {target_language} sentence (no source text, no language labels, no mixed output)."
        )
    else:
        instruction_lines.append("Keep the language exactly as spoken in the clip.")

    # 커스텀 프롬프트 추가
    if custom_prompt:
        instruction_lines.append(f"\nAdditional instructions: {custom_prompt}")
    whisper_windows = _build_whisper_windows(segment)
    if whisper_windows:
        instruction_lines.append("Use the following Whisper-derived segments (seconds from this clip's start). You may merge or split slightly, but do not exceed 10 seconds per subtitle entry:")
        instruction_lines.append(json.dumps(whisper_windows, ensure_ascii=False))
    else:
        speech_windows = _build_speech_windows(segment)
        if speech_windows:
            instruction_lines.append("Use the following speech windows (seconds from this clip's start, each <=10s) as guidance. You may merge or split them further for readability, but never allow a subtitle entry to exceed 10 seconds:")
            instruction_lines.append(json.dumps(speech_windows, ensure_ascii=False))
    whisper_context = _build_whisper_context(segment)
    if whisper_context:
        instruction_lines.append("Rely on the audio/video to confirm, but you may use this rough Whisper transcript as additional context:")
        instruction_lines.append(whisper_context)
    instruction_lines.append("Do not include any additional fields.")
    prompt = "\n".join(instruction_lines)

    response = _send_gemini_with_retry(
        client,
        job=job,
        context=f"{segment.index}번 세그먼트 전사",
        send_kwargs={
            "message": prompt,
            "response_schema": TRANSCRIPT_RESPONSE_SCHEMA,
            "media_paths": segment.file_path,
            "max_wait_seconds": 1200,
        },
    )
    payload = json.loads(response or "{}")
    segments = payload.get("segments", [])
    normalized: List[Dict[str, Any]] = []
    for item in segments:
        try:
            start = max(0.0, float(item.get("start", 0.0)))
            end = max(start, float(item.get("end", start + 0.1)))
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            normalized.append(
                {
                    "start": start + segment.start_time,
                    "end": end + segment.start_time,
                    "text": text,
                }
            )
        except (TypeError, ValueError):
            continue
    description = f"{len(normalized)}개 문장이 감지되었습니다."
    return normalized, description


def start_job(
    *,
    youtube_url: Optional[str],
    uploaded_file: Optional["FileStorage"],
    srt_file: Optional["FileStorage"],
    chunk_minutes: float,
    mode: str,
    target_language: Optional[str],
    custom_prompt: Optional[str],
    keep_original_entries: bool = False,
) -> Dict[str, Any]:
    if chunk_minutes <= 0:
        raise ValueError("청크 길이는 0보다 커야 합니다.")
    if mode not in {"transcribe", "translate"}:
        raise ValueError("지원하지 않는 전사 모드입니다.")
    if mode == "translate" and not target_language:
        raise ValueError("번역 모드에서는 번역 대상 언어가 필요합니다.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Google API Key가 설정되지 않았습니다.")

    # 원본 엔트리 유지 모드: SRT 파일 + 영상 필요
    if keep_original_entries:
        if not (srt_file and srt_file.filename):
            raise ValueError("원본 엔트리 유지 모드에서는 SRT 파일이 필요합니다.")
        if not (youtube_url or (uploaded_file and uploaded_file.filename)):
            raise ValueError("원본 엔트리 유지 모드에서는 영상 파일도 필요합니다.")
        if mode != "translate":
            raise ValueError("원본 엔트리 유지 모드는 번역 모드에서만 사용할 수 있습니다.")
    else:
        # 일반 모드: 영상이 필요
        if not (youtube_url or (uploaded_file and uploaded_file.filename)):
            raise ValueError("영상 파일 또는 YouTube 링크 중 하나가 필요합니다.")
        if srt_file and srt_file.filename and not (uploaded_file and uploaded_file.filename):
            raise ValueError("SRT 파일을 사용하려면 영상 파일도 함께 업로드해야 합니다.")

    job_id = uuid.uuid4().hex
    output_dir = _ensure_dir(os.path.join(SUBTITLE_JOB_ROOT, job_id))
    chunks_dir = _ensure_dir(os.path.join(output_dir, "segments"))
    uploaded_path: Optional[str] = None
    srt_path: Optional[str] = None
    if uploaded_file and uploaded_file.filename:
        uploaded_path = _save_uploaded_file(uploaded_file, os.path.join(output_dir, "source"))
    if srt_file and srt_file.filename:
        srt_path = _save_uploaded_file(srt_file, os.path.join(output_dir, "source"))
    job = SubtitleJob(
        job_id=job_id,
        output_dir=output_dir,
        chunks_dir=chunks_dir,
        chunk_minutes=chunk_minutes,
        mode=mode,
        target_language=target_language,
        youtube_url=youtube_url,
        custom_prompt=custom_prompt,
        keep_original_entries=keep_original_entries,
    )

    with _LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        args=(job, youtube_url, uploaded_path, srt_path),
        daemon=True,
    )
    thread.start()

    return job.to_dict()


def _translate_srt_entries_with_video(job: SubtitleJob, srt_path: str, segments_metadata: List[SegmentMetadata]) -> None:
    """원본 SRT 엔트리의 타임스탬프를 유지하면서 chunk 단위로 영상을 보고 번역합니다."""
    job.append_log("원본 자막 엔트리를 유지하면서 영상 분석 기반 번역을 시작합니다.")

    # SRT 파일 읽기
    subtitles = srt_module.read_srt(srt_path)
    if not subtitles:
        raise ValueError("SRT 파일에 자막이 없습니다.")

    job.append_log(f"총 {len(subtitles)}개의 자막 엔트리를 {len(segments_metadata)}개 청크로 처리합니다.")

    # Gemini 클라이언트 초기화
    client = GeminiClient(model=DEFAULT_MODEL, thinking_budget=-1)
    client.start_chat()

    last_segment_start_at: Optional[float] = None

    # 각 세그먼트(청크) 처리
    for seg_idx, segment in enumerate(job.segments):
        if last_segment_start_at is not None:
            elapsed = max(0.0, time.time() - last_segment_start_at)
            if elapsed < 60.0:
                wait_seconds = 60.0 - elapsed
                job.append_log(f"쿼터 보호를 위해 {wait_seconds:.1f}초 대기 후 다음 세그먼트를 처리합니다.")
                time.sleep(wait_seconds)

        last_segment_start_at = time.time()

        segment.status = "processing"
        segment.message = "번역 중..."
        job.append_log(f"{segment.index}번 세그먼트 처리 중...")
        job.updated_at = _now()

        try:
            # 이 세그먼트 시간 범위에 속하는 자막들 찾기
            segment_subtitles = []
            for subtitle in subtitles:
                start_sec = _srt_timestamp_to_seconds(subtitle['start'])
                end_sec = _srt_timestamp_to_seconds(subtitle['end'])

                # 자막이 세그먼트 범위와 겹치는지 확인
                if start_sec < segment.end_time and end_sec > segment.start_time:
                    segment_subtitles.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': subtitle['text']
                    })

            if not segment_subtitles:
                segment.status = "completed"
                segment.message = "자막 없음"
                job.processed_segments += 1
                continue

            job.append_log(f"세그먼트 {segment.index}에 {len(segment_subtitles)}개 자막 포함")

            # 번역 프롬프트 구성
            instruction_lines = [
                "You will receive a video segment with subtitles.",
                f"Translate each subtitle into {job.target_language} while watching the video for context.",
                "Return a JSON array with the translated subtitles maintaining the exact same timestamps.",
                "Each entry should have: start (seconds), end (seconds), text (translated).",
            ]
            if job.custom_prompt:
                instruction_lines.append(f"\nAdditional instructions: {job.custom_prompt}")

            # 원본 자막 리스트 추가
            instruction_lines.append("\nOriginal subtitles:")
            for sub in segment_subtitles:
                instruction_lines.append(f"[{sub['start']:.2f}s - {sub['end']:.2f}s] {sub['text']}")

            prompt = "\n".join(instruction_lines)

            # Gemini에 영상과 함께 번역 요청
            response = _send_gemini_with_retry(
                client,
                job=job,
                context=f"{segment.index}번 세그먼트 번역",
                send_kwargs={
                    "message": prompt,
                    "media_paths": [segment.file_path],
                    "response_schema": {
                        "type": "object",
                        "properties": {
                            "subtitles": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start": {"type": "number"},
                                        "end": {"type": "number"},
                                        "text": {"type": "string"}
                                    },
                                    "required": ["start", "end", "text"]
                                }
                            }
                        },
                        "required": ["subtitles"]
                    },
                    "max_wait_seconds": 600,
                },
            )

            # 응답 파싱
            if isinstance(response, str):
                payload = json.loads(response or "{}")
            elif isinstance(response, dict):
                payload = response
            else:
                payload = {}
            translated_subs = payload.get('subtitles', [])

            # 번역된 자막들을 transcript_entries에 추가
            for trans_sub in translated_subs:
                job.transcript_entries.append({
                    "start": trans_sub['start'],
                    "end": trans_sub['end'],
                    "text": trans_sub['text']
                })

            segment.transcript = translated_subs
            segment.status = "completed"
            segment.message = f"{len(translated_subs)}개 자막 번역 완료"
            job.processed_segments += 1
            job.updated_at = _now()

            job.append_log(f"세그먼트 {segment.index} 번역 완료: {len(translated_subs)}개 자막")

        except Exception as exc:
            logger.exception(f"세그먼트 {segment.index} 처리 실패")
            job.append_log(f"세그먼트 {segment.index} 처리 실패: {exc}. 원본 텍스트를 사용합니다.", level="warning")

            # 실패 시 원본 텍스트 사용
            for sub in segment_subtitles:
                job.transcript_entries.append({
                    "start": sub['start'],
                    "end": sub['end'],
                    "text": sub['text']
                })

            segment.transcript = segment_subtitles
            segment.status = "completed"
            segment.message = "실패 (원본 사용)"
            job.processed_segments += 1
            job.updated_at = _now()


def _run_job(job: SubtitleJob, youtube_url: Optional[str], uploaded_path: Optional[str], srt_path: Optional[str] = None) -> None:
    try:
        # 원본 엔트리 유지 모드: SRT 타임스탬프 유지하면서 chunk 단위로 번역
        if job.keep_original_entries and srt_path:
            _update_job(job, status="running", phase="source", message="영상 소스를 준비하는 중입니다.")
            job.source_path = _prepare_source_path(job, youtube_url, uploaded_path)
            job.append_log(f"소스 파일: {job.source_path}")

            _update_job(job, phase="split", message="영상을 청크 단위로 분할합니다.")
            ffmpeg_module.register_ffmpeg_path()

            # SRT 기반으로 세그먼트 생성
            segments_metadata = _create_segments_from_srt(job, srt_path)
            job.total_segments = len(segments_metadata)

            # 세그먼트 상태 초기화
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

            _update_job(job, phase="translate", message="원본 자막 엔트리를 유지하면서 영상을 분석하여 번역합니다.")
            _translate_srt_entries_with_video(job, srt_path, segments_metadata)

            # SRT 파일 저장
            combined = sorted(job.transcript_entries, key=lambda item: item["start"])
            srt_output_path = os.path.join(job.output_dir, f"{job.job_id}.srt")
            job.transcript_path = _write_srt(combined, srt_output_path)
            _update_job(job, status="completed", phase="finished", message="번역이 완료되었습니다.")
            return

        # 기본 모드: 영상 분할 및 전사/번역
        _update_job(job, status="running", phase="source", message="영상 소스를 준비하는 중입니다.")
        job.source_path = _prepare_source_path(job, youtube_url, uploaded_path)
        job.append_log(f"소스 파일: {job.source_path}")

        _update_job(job, phase="split", message="영상 분할을 시작합니다.")
        ffmpeg_module.register_ffmpeg_path()

        # SRT 파일이 제공된 경우 VAD/Whisper를 건너뛰고 SRT 기반으로 세그먼트 생성
        if srt_path:
            job.append_log(f"업로드된 SRT 파일을 사용하여 세그먼트를 생성합니다: {srt_path}")
            segments_metadata = _create_segments_from_srt(job, srt_path)
        else:
            segments_metadata = _create_segments(job, job.chunk_minutes)

        job.total_segments = len(segments_metadata)

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

        _update_job(job, phase="transcribe", message="Gemini로 전사를 시작합니다.")
        client = GeminiClient(model=DEFAULT_MODEL, thinking_budget=-1)
        client.start_chat()
        last_segment_start_at: Optional[float] = None

        for segment in job.segments:
            if last_segment_start_at is not None:
                elapsed = max(0.0, time.time() - last_segment_start_at)
                if elapsed < 60.0:
                    wait_seconds = 60.0 - elapsed
                    job.append_log(f"쿼터 보호를 위해 {wait_seconds:.1f}초 대기 후 다음 세그먼트를 처리합니다.")
                    time.sleep(wait_seconds)

            last_segment_start_at = time.time()

            segment.status = "processing"
            segment.message = "전사 중..."
            job.append_log(f"{segment.index}번 세그먼트 전사 중...")
            job.updated_at = _now()

            # SRT 파일이 제공된 경우 Whisper 단계를 건너뜀
            if not srt_path:
                try:
                    segment.whisper_segments = _generate_whisper_segments(segment)
                    if segment.whisper_segments:
                        job.append_log(
                            f"Whisper 참고 세그먼트 {len(segment.whisper_segments)}개를 확보했습니다."
                        )
                except Exception as whisper_exc:
                    segment.whisper_segments = []
                    job.append_log(f"Whisper 기반 세그먼트 생성 실패: {whisper_exc}")
            else:
                segment.whisper_segments = []
                job.append_log("SRT 파일이 제공되어 Whisper 단계를 건너뜁니다.")

            normalized, description = _transcribe_segment(
                job, client, segment, job.mode, job.target_language, job.custom_prompt
            )
            segment.status = "completed"
            segment.message = description
            segment.transcript = normalized
            job.transcript_entries.extend(normalized)

            job.processed_segments += 1
            job.updated_at = _now()
            context_text = _build_transcript_context(job.transcript_entries)
            if context_text:
                history_seed = [{'role': 'user', 'parts': [{'text': context_text}]}]
            else:
                history_seed = []
            client.start_chat(history=history_seed, suppress_log=True, reset_usage=False)

        combined = sorted(job.transcript_entries, key=lambda item: item["start"])
        if not combined:
            job.append_log("전사된 문장을 찾지 못했습니다.")
        srt_path = os.path.join(job.output_dir, f"{job.job_id}.srt")
        job.transcript_path = _write_srt(combined, srt_path)
        _update_job(job, status="completed", phase="finished", message="전사 작업이 완료되었습니다.")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Subtitle job %s failed", job.job_id)
        _update_job(job, status="failed", phase="error", message="작업 중 오류가 발생했습니다.", error=str(exc))


__all__ = [
    "start_job",
    "get_job_data",
    "get_transcript_path",
    "get_segment_path",
]
