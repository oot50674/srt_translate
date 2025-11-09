"""FFmpeg 실행 파일 경로 등록 유틸리티.

이 모듈은 배포 번들에 포함된 FFmpeg 바이너리(`ffmpeg-8.0/bin`)를
찾아 환경 변수에 등록합니다. 경로 등록은 모듈 임포트 시점에
자동으로 한 번 수행되며, 필요 시 `register_ffmpeg_path()`를 수동으로
호출해 재등록할 수 있습니다.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional
import subprocess
import math
import argparse
from yt_dlp import YoutubeDL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_project_root(base_path: Optional[str] = None) -> str:
    """프로젝트 루트 경로를 반환합니다."""
    if base_path:
        root = os.path.abspath(base_path)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(current_dir, os.pardir))

    ffmpeg_dir = os.path.join(root, "ffmpeg-8.0")
    if os.path.isdir(ffmpeg_dir):
        return root

    parent = os.path.dirname(root)
    if os.path.isdir(os.path.join(parent, "ffmpeg-8.0")):
        return parent

    return root


def resolve_ffmpeg_bin_dir(base_path: Optional[str] = None) -> str:
    """FFmpeg 바이너리가 위치한 bin 디렉터리를 반환합니다."""
    project_root = _get_project_root(base_path)
    candidate = os.path.join(project_root, "ffmpeg-8.0", "bin")
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"FFmpeg bin directory not found: {candidate}")

def _binary_name(base: str) -> str:
    """바이너리 이름을 반환합니다."""
    return f"{base}.exe" if os.name == "nt" else base


def register_ffmpeg_path(base_path: Optional[str] = None) -> Dict[str, str]:
    """FFmpeg 및 FFprobe 실행 파일 경로를 환경 변수에 등록합니다."""
    bin_dir = resolve_ffmpeg_bin_dir(base_path)
    ffmpeg_binary = os.path.join(bin_dir, _binary_name("ffmpeg"))
    ffprobe_binary = os.path.join(bin_dir, _binary_name("ffprobe"))

    if not os.path.isfile(ffmpeg_binary):
        raise FileNotFoundError(f"ffmpeg binary not found: {ffmpeg_binary}")
    if not os.path.isfile(ffprobe_binary):
        raise FileNotFoundError(f"ffprobe binary not found: {ffprobe_binary}")

    # PATH에 bin 디렉터리가 없으면 추가
    current_path = os.environ.get("PATH", "")
    path_parts = [p for p in current_path.split(os.pathsep) if p]
    if bin_dir not in path_parts:
        new_path = os.pathsep.join([bin_dir] + path_parts)
        os.environ["PATH"] = new_path
        logger.info("FFmpeg bin directory added to PATH: %s", bin_dir)
    else:
        logger.debug("FFmpeg bin directory already in PATH: %s", bin_dir)

    # 추가 호환성을 위한 환경 변수 설정
    os.environ["FFMPEG_PATH"] = ffmpeg_binary
    os.environ["FFPROBE_PATH"] = ffprobe_binary
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_binary

    logger.info("FFmpeg executables registered: ffmpeg=%s, ffprobe=%s", ffmpeg_binary, ffprobe_binary)
    return {
        "ffmpeg": ffmpeg_binary,
        "ffprobe": ffprobe_binary,
    }


try:
    REGISTERED_FFMPEG = register_ffmpeg_path()
except FileNotFoundError as exc:
    REGISTERED_FFMPEG = {}
    logger.warning("FFmpeg registration skipped: %s", exc)


__all__ = ["register_ffmpeg_path", "resolve_ffmpeg_bin_dir", "REGISTERED_FFMPEG"]


def get_stream_url(youtube_url: str) -> str:
    """유튜브 URL에서 직접 스트림 URL을 얻습니다."""
    ydl_opts = {"quiet": True, "format": "bv*[ext=mp4]/bv*+ba/b"}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        if "url" in info:
            return info["url"]
        for f in sorted(info.get("formats", []), key=lambda x: x.get("height", 0), reverse=True):
            if f.get("vcodec") not in (None, "none") and f.get("url"):
                return f["url"]
    raise RuntimeError("비디오 스트림 URL을 찾지 못했습니다.")


def get_duration_seconds(input_url_or_path: str) -> float:
    """ffprobe를 사용해 동영상 길이를 초 단위로 구합니다."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_url_or_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    return float(out)


def make_even_timestamps(duration: float, count: int, margin: float = 0.2) -> list[float]:
    """지정된 시간 동안 균등하게 분배된 타임스탬프 목록을 생성합니다."""
    start = max(0.0, margin)
    end = max(start + margin, duration - margin)
    if count <= 0:
        return []
    return [start + (end - start) * (i + 1) / (count + 1) for i in range(count)]


def snapshot_at_times(input_url_or_path: str, timestamps: list[float], out_dir: str, prefix: str = "snap"):
    """지정된 타임스탬프에서 스냅샷을 추출합니다."""
    os.makedirs(out_dir, exist_ok=True)
    for idx, t in enumerate(timestamps, 1):
        out_path = os.path.join(out_dir, f"{prefix}_{idx:03d}.jpg")
        cmd = [
            "ffmpeg",
            "-ss", f"{t:.3f}",
            "-i", input_url_or_path,
            "-frames:v", "1",
            "-y",
            out_path,
        ]
        subprocess.check_call(cmd)


def snapshots_from_youtube(youtube_url: str, count: int, out_dir: str, margin: float = 0.2):
    """유튜브 영상에서 스냅샷을 추출합니다."""
    
    # 유튜브 URL에서 직접 스트림 URL 추출
    stream_url = get_stream_url(youtube_url)
    # 동영상 전체 길이(초) 확인
    duration = get_duration_seconds(stream_url)
    # 균등 분배된 타임스탬프 생성
    ts = make_even_timestamps(duration, count, margin=margin)
    # 각 타임스탬프에서 스냅샷 추출
    snapshot_at_times(stream_url, ts, out_dir)
    # 생성된 스냅샷 파일 경로 목록 반환
    return [os.path.join(out_dir, f"snap_{i+1:03d}.jpg") for i in range(len(ts))]

# __all__ 업데이트
__all__.extend([
    "get_stream_url", "get_duration_seconds", "make_even_timestamps",
    "snapshot_at_times", "snapshots_from_youtube"
])


def _has_stream(input_path: str, selector: str) -> bool:
    """지정한 타입(selector)의 스트림이 존재하는지 확인합니다."""
    register_ffmpeg_path()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        selector,
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        input_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.decode().strip())


def has_video_stream(input_path: str) -> bool:
    """비디오 스트림 존재 여부를 반환합니다."""
    return _has_stream(input_path, "v")


def has_audio_stream(input_path: str) -> bool:
    """오디오 스트림 존재 여부를 반환합니다."""
    return _has_stream(input_path, "a")


__all__.extend(["has_video_stream", "has_audio_stream"])

if __name__ == "__main__":
    # 프로젝트 루트를 기준으로 기본 출력 디렉토리 설정
    try:
        project_root = _get_project_root()
        default_out_dir = os.path.join(project_root, "my_snapshots")
    except Exception:
        # 프로젝트 루트를 찾지 못할 경우 현재 디렉토리에 저장
        default_out_dir = "my_snapshots"

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--count", type=int, default=5, help="number of snapshots")
    parser.add_argument("--out", default=default_out_dir, help="output dir")
    args = parser.parse_args()

    register_ffmpeg_path()
    snapshots = snapshots_from_youtube(args.url, args.count, args.out)
    print("Saved:", *snapshots, sep="\n")

