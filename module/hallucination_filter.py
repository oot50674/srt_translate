from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# segment_analyzer removed; use only _is_suspicious_length for candidate
# selection/verification
from module.ffmpeg_module import register_ffmpeg_path
from module.Whisper_util import WhisperUtil, get_whisper_util

logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 0.3
MIN_TEXT_LENGTH = 0
CHARS_PER_SECOND_THRESHOLD = 12
LONG_HALLUCINATION_DURATION_SEC = 20.0
MIN_ABSOLUTE_TEXT_CHARS = 200

def _is_suspicious_length(
    text: str,
    duration: float,
    *,
    min_text_length: int = MIN_TEXT_LENGTH,
    min_chars_per_sec: float = CHARS_PER_SECOND_THRESHOLD,
) -> bool:
    """길이에 비해 문장이 너무 긴 경우만 환각 후보로 삼습니다."""
    if duration < MIN_DURATION_SEC:
        return False
    stripped = text.strip()
    if min_text_length > 0 and len(stripped) < min_text_length:
        return False
    chars_per_sec = len(stripped) / max(duration, 0.001)
    return chars_per_sec >= min_chars_per_sec


def _is_normal_entry(
    text: str,
    duration: float,
    *,
    min_text_length: int = MIN_TEXT_LENGTH,
    min_chars_per_sec: float = CHARS_PER_SECOND_THRESHOLD,
) -> bool:
    """모든 조건을 통과할 때만 정상 엔트리로 간주합니다."""
    stripped = text.strip()
    if len(stripped) >= MIN_ABSOLUTE_TEXT_CHARS:
        return False
    if duration >= LONG_HALLUCINATION_DURATION_SEC:
        return False
    return not _is_suspicious_length(
        stripped,
        duration,
        min_text_length=min_text_length,
        min_chars_per_sec=min_chars_per_sec,
    )


def _extract_audio_clip(source_path: str, start: float, end: float) -> Optional[str]:
    """지정 구간의 오디오만 추출한 임시 파일을 반환합니다."""
    register_ffmpeg_path()
    start = max(0.0, float(start))
    end = max(start + 0.05, float(end))
    if not os.path.isfile(source_path):
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        clip_path = tmp.name
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        clip_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return clip_path
    except subprocess.CalledProcessError as exc:
        logger.warning("오디오 클립 추출 실패: %s", exc)
        return None


def _retranscribe_clip(
    source_path: str,
    start: float,
    end: float,
    whisper: Optional[WhisperUtil],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    clip_path = _extract_audio_clip(source_path, start, end)
    if not clip_path:
        return None, []
    whisper_util = whisper or get_whisper_util()
    try:
        result = whisper_util.transcribe_audio(clip_path, show_progress=False)
        segments = result.get("segments") or []
        texts = [str(seg.get("text", "")).strip() for seg in segments if str(seg.get("text", "")).strip()]
        if texts:
            return " ".join(texts).strip(), segments
        return str(result.get("text", "")).strip(), segments
    except Exception as exc:  # pragma: no cover - 런타임 방어
        logger.warning("클립 재전사 실패(start=%.3f, end=%.3f): %s", start, end, exc)
        return None, []
    finally:
        try:
            os.remove(clip_path)
        except OSError:
            pass


def fix_repetitive_hallucinations(
    entries: List[Dict[str, Any]],
    source_path: str,
    whisper: Optional[WhisperUtil] = None,
    *,
    min_chars_per_second: float = CHARS_PER_SECOND_THRESHOLD,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    길이에 비해 과도하게 긴 자막을 길이/재생시간(문자/초) 기준으로 재전사/표시합니다.

    Returns:
        (entries, stats) 튜플. stats는 재전사/표시 건수를 포함합니다.
    """
    stats = {
        "candidates": 0,
        "retranscribed": 0,
        "succeeded": 0,
        "flagged": 0,
    }
    if not entries or not source_path:
        return entries, stats

    processed_entries: List[Dict[str, Any]] = []

    for entry in entries:
        try:
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", start))
        except (TypeError, ValueError):
            processed_entries.append(entry)
            continue
        duration = max(0.0, end - start)
        text = str(entry.get("text", "")).strip()
        stripped_text = text.strip()
        is_long_duration = duration >= LONG_HALLUCINATION_DURATION_SEC
        is_long_text = len(stripped_text) >= MIN_ABSOLUTE_TEXT_CHARS
        is_suspicious = _is_suspicious_length(
            text,
            duration,
            min_text_length=min_text_length,
            min_chars_per_sec=min_chars_per_second,
        )
        # 후보: 문자/초 비율이 이상하거나(기존), 절대 길이(>=MIN_ABSOLUTE_TEXT_CHARS), 또는 절대 길이(>=LONG_HALLUCINATION_DURATION_SEC)
        if not is_suspicious and not is_long_duration and not is_long_text:
            processed_entries.append(entry)
            continue
        # 길이/재생시간 비율 또는 절대 길이(>=10초) 조건으로만 후보를 고릅니다.

        stats["candidates"] += 1
        entry.setdefault("original_text", text)

        new_text, _segments = _retranscribe_clip(source_path, start, end, whisper)
        stats["retranscribed"] += 1
        if new_text:
            stripped_new = new_text.strip()
            # 재전사 후 추가 검사 없이 바로 채택
            entry["text"] = stripped_new
            stats["succeeded"] += 1
            processed_entries.append(entry)
            continue
        # 재전사 실패 시 원문 유지하고 플래그만 증가
        stats["flagged"] += 1
        processed_entries.append(entry)

    return processed_entries, stats
