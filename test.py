import argparse
import os
import shutil
import sys
import tempfile
from typing import List

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)

from module import ffmpeg_module
from module.gemini_module import GeminiClient


def fetch_snapshots(youtube_url: str, count: int) -> List[str]:
    """Download evenly spaced snapshots for the given YouTube URL."""
    temp_dir = tempfile.mkdtemp(prefix="yt_cli_snap_")
    try:
        snapshots = ffmpeg_module.snapshots_from_youtube(
            youtube_url,
            count=count,
            out_dir=temp_dir,
        )
        if not snapshots:
            raise RuntimeError("스냅샷 생성에 실패했습니다.")
        return snapshots
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def analyze_snapshots(youtube_url: str, count: int, prompt: str, keep: bool) -> None:
    """Create snapshots and request a multimodal summary from Gemini."""
    print(f"[1/3] 유튜브 링크에서 스냅샷 {count}장을 추출합니다...")
    snapshots = fetch_snapshots(youtube_url, count)

    print(f"       생성된 스냅샷 ({len(snapshots)}장):")
    for path in snapshots:
        print(f"         - {path}")

    client = GeminiClient()
    client.start_chat()

    print("[2/3] Gemini에 이미지 분석을 요청합니다...")
    response = client.send_message(prompt, media_paths=snapshots)

    print("[3/3] 모델 응답:")
    print("-" * 60)
    print(response or "[응답이 비어 있습니다]")
    print("-" * 60)

    if keep:
        print("임시 스냅샷 디렉터리를 보존합니다.")
    else:
        temp_dir = os.path.dirname(snapshots[0])
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("임시 스냅샷 디렉터리를 삭제했습니다.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YouTube 영상으로부터 스냅샷을 생성하고 Gemini가 이미지를 읽는지 확인하는 CLI 도구",
    )
    parser.add_argument("youtube_url", help="분석할 YouTube 영상 URL")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="추출할 스냅샷 수 (기본값: 5)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "이 이미지들은 번역 예정인 유튜브 영상의 장면입니다. "
            "장소, 등장인물, 분위기, 주요 사건을 요약하고 "
            "번역 시 주의해야 할 맥락 정보를 정리해 주세요."
        ),
        help="이미지 분석용 프롬프트",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="생성된 스냅샷 파일을 삭제하지 않고 유지합니다.",
    )
    return parser


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    try:
        analyze_snapshots(
            youtube_url=args.youtube_url,
            count=args.count,
            prompt=args.prompt,
            keep=args.keep,
        )
        return 0
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"오류 발생: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
