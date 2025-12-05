"""Silero VAD 를 이용해 비디오를 발화 구간 단위로 분할하는 모듈."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence, Optional

from module.cuda_runtime import register_embedded_cuda

register_embedded_cuda()

import torch

from module.ffmpeg_module import (
    get_duration_seconds,
    register_ffmpeg_path,
    has_video_stream,
    has_audio_stream,
)
from module.storage import get_value as storage_get_value
from module.storage import set_value as storage_set_value


DEFAULT_STORAGE_KEY = "video_segments"


from module.silero_vad import SileroVAD


@dataclass(slots=True)
class SegmentMetadata:
    """분할된 각 비디오 조각의 메타데이터."""

    index: int
    file_path: str
    start_time: float
    end_time: float
    duration: float
    speech_segments: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, int | float | str | List[Dict[str, float]]]:
        """dict 형태로 직렬화합니다."""
        return asdict(self)


def _validate_inputs(input_path: str, output_dir: str, minutes_per_segment: float) -> None:
    if minutes_per_segment <= 0:
        raise ValueError("minutes_per_segment는 0보다 커야 합니다.")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 비디오를 찾을 수 없습니다: {input_path}")
    os.makedirs(output_dir, exist_ok=True)


def _extract_audio_for_vad(input_path: str, sampling_rate: int = 16000) -> str:
    register_ffmpeg_path()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"오디오 추출 실패: {stderr.strip()}") from exc
    return tmp_path


def _cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _clamp_segments(
    speech_segments: Sequence[Dict[str, float]],
    max_duration: float,
) -> List[Dict[str, float]]:
    clamped: List[Dict[str, float]] = []
    for seg in speech_segments:
        start = max(0.0, float(seg.get("start", 0.0)))
        end = min(max_duration, float(seg.get("end", 0.0)))
        if end <= start:
            continue
        clamped.append({"start": round(start, 3), "end": round(end, 3)})
    clamped.sort(key=lambda seg: seg["start"])
    return clamped


def _group_speech_segments(
    speech_segments: Sequence[Dict[str, float]],
    max_chunk_duration: float,
) -> List[Dict[str, object]]:
    if not speech_segments:
        return []

    grouped: List[Dict[str, object]] = []
    current_segments: List[Dict[str, float]] = []
    current_start = speech_segments[0]["start"]
    current_end = speech_segments[0]["end"]

    for seg in speech_segments:
        start = seg["start"]
        end = seg["end"]
        if not current_segments:
            current_segments = [seg]
            current_start = start
            current_end = end
            continue

        projected_duration = end - current_start
        # 같은 청크에 추가할지 여부 판단 (단, 단일 발화가 목표 길이를 초과하면 그대로 허용)
        if projected_duration <= max_chunk_duration or (end - start) >= max_chunk_duration:
            current_segments.append(seg)
            current_end = end
        else:
            grouped.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "segments": current_segments,
                }
            )
            current_segments = [seg]
            current_start = start
            current_end = end

    if current_segments:
        grouped.append(
            {
                "start": current_start,
                "end": current_end,
                "segments": current_segments,
            }
        )

    return grouped


def _generate_output_path(output_dir: str, prefix: str, index: int, extension: str) -> str:
    return os.path.join(output_dir, f"{prefix}_{index:03d}{extension}")


def _detect_speech_segments(
    input_path: str,
    *,
    vad_threshold: float,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 250,
    speech_pad_ms: int = 30,
) -> tuple[List[Dict[str, float]], float, bool, str]:
    """파일에서 발화 구간을 추출하고 기본 메타 정보를 반환합니다."""

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 비디오를 찾을 수 없습니다: {input_path}")

    video_duration = get_duration_seconds(input_path)
    source_has_video = has_video_stream(input_path)
    source_has_audio = has_audio_stream(input_path)
    if not source_has_audio:
        raise ValueError("오디오 트랙이 없는 미디어는 분할할 수 없습니다.")

    audio_path = _extract_audio_for_vad(input_path)
    try:
        vad = SileroVAD(
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        speech_segments = vad.detect_speech_from_file(audio_path)
    finally:
        _cleanup_temp_file(audio_path)

    speech_segments = _clamp_segments(speech_segments, video_duration)
    extension = os.path.splitext(input_path)[1] or ".mp4"
    return speech_segments, video_duration, source_has_video, extension


def _cut_media_chunk(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
    *,
    include_video: bool,
) -> None:
    register_ffmpeg_path()
    duration = max(end - start, 0.1)
    if include_video:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-reset_timestamps",
            "1",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "copy",
            "-vn",
            output_path,
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"미디어 분할 실패: {stderr.strip()}") from exc


def _store_metadata(segments: Iterable[SegmentMetadata], storage_key: str) -> None:
    storage_set_value(storage_key, [segment.to_dict() for segment in segments])


def split_video_by_minutes(
    input_path: str,
    output_dir: str,
    minutes_per_segment: float = 1.0,
    storage_key: str = DEFAULT_STORAGE_KEY,
    prefix: str = "segment",
    vad_threshold: float = 0.7,
) -> List[SegmentMetadata]:
    """Silero VAD 결과를 기반으로 분 단위 내에서 비디오를 분할합니다."""

    _validate_inputs(input_path, output_dir, minutes_per_segment)
    speech_segments, video_duration, source_has_video, extension = _detect_speech_segments(
        input_path,
        vad_threshold=vad_threshold,
    )
    max_duration = max(1.0, minutes_per_segment * 60.0)

    if not speech_segments:
        # 발화가 감지되지 않으면 전체 비디오를 하나의 구간으로 처리합니다.
        grouped_segments = [
            {"start": 0.0, "end": round(video_duration, 3), "segments": []}
        ]
    else:
        grouped_segments = _group_speech_segments(speech_segments, max_duration)

    metadata: List[SegmentMetadata] = []
    for idx, group in enumerate(grouped_segments, start=1):
        start = float(group["start"])
        end = float(group["end"])
        if end <= start:
            continue

        output_path = _generate_output_path(output_dir, prefix, idx, extension)
        _cut_media_chunk(
            input_path,
            output_path,
            start,
            end,
            include_video=source_has_video,
        )
        metadata.append(
            SegmentMetadata(
                index=idx,
                file_path=os.path.abspath(output_path),
                start_time=round(start, 3),
                end_time=round(end, 3),
                duration=round(end - start, 3),
                speech_segments=[dict(seg) for seg in group.get("segments", [])],
            )
        )

    _store_metadata(metadata, storage_key)
    return metadata


def split_video_by_utterances(
    input_path: str,
    output_dir: str,
    *,
    storage_key: str = DEFAULT_STORAGE_KEY,
    prefix: str = "utterance",
    vad_threshold: float = 0.6,
) -> List[SegmentMetadata]:
    """발화 구간을 그대로 사용하여 비디오/오디오를 잘라냅니다."""

    MIN_UTTERANCE_DURATION = 0.5  # 이하 발화는 전사에서 제외
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"입력 비디오를 찾을 수 없습니다: {input_path}")
    os.makedirs(output_dir, exist_ok=True)

    speech_segments, video_duration, source_has_video, extension = _detect_speech_segments(
        input_path,
        vad_threshold=vad_threshold,
        min_speech_duration_ms=200,
        min_silence_duration_ms=600,
        speech_pad_ms=120,
    )

    # 발화가 없으면 전체 파일을 하나의 세그먼트로 처리
    if not speech_segments:
        speech_segments = [{"start": 0.0, "end": round(video_duration, 3)}]

    metadata: List[SegmentMetadata] = []
    for idx, seg in enumerate(speech_segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue
        if (end - start) <= MIN_UTTERANCE_DURATION:
            # 너무 짧은 발화는 건너뛴다.
            continue
        output_path = _generate_output_path(output_dir, prefix, idx, extension)
        _cut_media_chunk(
            input_path,
            output_path,
            start,
            end,
            include_video=source_has_video,
        )
        metadata.append(
            SegmentMetadata(
                index=idx,
                file_path=os.path.abspath(output_path),
                start_time=round(start, 3),
                end_time=round(end, 3),
                duration=round(end - start, 3),
                speech_segments=[{"start": round(start, 3), "end": round(end, 3)}],
            )
        )

    _store_metadata(metadata, storage_key)
    return metadata


def get_stored_segments(storage_key: str = DEFAULT_STORAGE_KEY) -> List[dict]:
    """storage에 저장된 분할 메타데이터를 반환합니다."""

    stored = storage_get_value(storage_key, default=[])
    return list(stored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Silero VAD 기반 비디오 분할기.")
    parser.add_argument("input", help="입력 비디오 경로")
    parser.add_argument("output_dir", help="분할 파일을 저장할 폴더")
    parser.add_argument(
        "--minutes",
        type=float,
        default=1.0,
        help="목표 분할 길이(분 단위, 기본 1분)",
    )
    parser.add_argument(
        "--storage-key",
        default=DEFAULT_STORAGE_KEY,
        help="storage.py에 저장할 키 이름",
    )
    parser.add_argument(
        "--prefix",
        default="segment",
        help="출력 파일 접두어 (기본 segment)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.7,
        help="Silero VAD 임계값 (0.35~0.7 권장)",
    )
    args = parser.parse_args()

    metadata = split_video_by_minutes(
        input_path=args.input,
        output_dir=args.output_dir,
        minutes_per_segment=args.minutes,
        storage_key=args.storage_key,
        prefix=args.prefix,
        vad_threshold=args.vad_threshold,
    )
    print(f"{len(metadata)}개의 비디오 조각을 생성했습니다.")
    for segment in metadata:
        print(
            f"[{segment.index:02d}] {segment.file_path} "
            f"({segment.start_time:.3f}s ~ {segment.end_time:.3f}s)"
        )
