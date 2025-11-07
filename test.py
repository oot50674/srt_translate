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
from module.Whisper_util import (
    WhisperUtil,
    get_whisper_util,
    transcribe_audio_file,
    transcribe_audio_with_timestamps,
)


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
        description="YouTube 영상 분석 및 Whisper 음성 인식 테스트 CLI 도구",
    )

    subparsers = parser.add_subparsers(dest="command", help="사용할 기능 선택")

    # YouTube 분석 서브커맨드
    yt_parser = subparsers.add_parser("youtube", help="YouTube 영상 분석")
    yt_parser.add_argument("youtube_url", help="분석할 YouTube 영상 URL")
    yt_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="추출할 스냅샷 수 (기본값: 5)",
    )
    yt_parser.add_argument(
        "--prompt",
        default=(
            "이 이미지들은 번역 예정인 유튜브 영상의 장면입니다. "
            "장소, 등장인물, 분위기, 주요 사건을 요약하고 "
            "번역 시 주의해야 할 맥락 정보를 정리해 주세요."
        ),
        help="이미지 분석용 프롬프트",
    )
    yt_parser.add_argument(
        "--keep",
        action="store_true",
        help="생성된 스냅샷 파일을 삭제하지 않고 유지합니다.",
    )

    # Whisper 테스트 서브커맨드
    whisper_parser = subparsers.add_parser("whisper", help="Whisper 음성 인식 모듈 테스트")

    return parser


def test_whisper_model_loading():
    """Whisper 모델 로딩 테스트"""
    print("Whisper 모델 로딩 테스트 시작...")
    try:
        util = WhisperUtil()
        print("모델 로딩 성공")
        return True
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return False


def test_whisper_singleton():
    """Whisper 싱글톤 패턴 테스트"""
    print("Whisper 싱글톤 패턴 테스트 시작...")
    try:
        util1 = get_whisper_util()
        util2 = get_whisper_util()

        if util1 is util2:
            print("싱글톤 패턴 작동 확인")
            return True
        else:
            print("싱글톤 패턴 실패")
            return False
    except Exception as e:
        print(f"싱글톤 테스트 실패: {e}")
        return False


def test_whisper_file_not_found():
    """존재하지 않는 파일로 에러 처리 테스트"""
    print("Whisper 파일 존재하지 않을 때 에러 처리 테스트 시작...")
    try:
        # 존재하지 않는 파일로 테스트 (에러 처리 확인)
        dummy_path = "nonexistent_audio.mp3"
        try:
            result = transcribe_audio_file(dummy_path)
            print("존재하지 않는 파일인데 변환 성공 (비정상)")
            return False
        except FileNotFoundError:
            print("파일 존재하지 않을 때 적절한 에러 처리")
            return True
        except Exception as e:
            print(f"예상치 못한 에러: {e}")
            return False
    except Exception as e:
        print(f"파일 존재 테스트 실패: {e}")
        return False


def test_whisper_with_mp4_file():
    """Feet.mp4 파일로 실제 변환 테스트"""
    print("MP4 파일 실제 변환 테스트 시작...")
    try:
        mp4_path = "download_video/Feet.mp4"

        if not os.path.exists(mp4_path):
            print(f"테스트 파일이 존재하지 않습니다: {mp4_path}")
            return False

        print(f"테스트 파일: {mp4_path}")
        file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB 단위
        print(f"파일 크기: {file_size:.1f} MB")
        print("진행 상황 표시: 활성화")
        # 언어 자동 감지 모드 + 진행 상황 표시로 테스트
        result_text = transcribe_audio_file(mp4_path, language=None, show_progress=True)

        if result_text and len(result_text.strip()) > 0:
            print("MP4 파일 변환 성공")
            print(f"변환된 텍스트 길이: {len(result_text)} 문자")
            print(f"변환된 텍스트 (처음 200자): {result_text[:200]}...")
            return True
        else:
            print("변환 결과가 비어있음")
            return False

    except Exception as e:
        print(f"MP4 파일 변환 테스트 실패: {e}")
        return False


def test_whisper_util_methods():
    """WhisperUtil 메소드들 테스트"""
    print("WhisperUtil 메소드 테스트 시작...")
    try:
        util = get_whisper_util()

        # 메소드 존재 확인
        if hasattr(util, 'transcribe_audio') and hasattr(util, 'get_text_only'):
            print("주요 메소드 존재 확인")
        else:
            print("주요 메소드 누락")
            return False

        # 모델 로드 확인
        if util.model is not None:
            print("모델 인스턴스 존재 확인")
        else:
            print("모델 인스턴스 없음")
            return False

        # 디바이스 설정 확인
        if util.device == "cpu":
            print("CPU 전용 설정 확인")
        else:
            print("CPU 설정이 아님")
            return False

        return True
    except Exception as e:
        print(f"메소드 테스트 실패: {e}")
        return False


def test_whisper_progress_display():
    """진행 상황 표시 기능 테스트"""
    print("진행 상황 표시 기능 테스트 시작...")
    try:
        # 짧은 파일로 빠르게 테스트 (진행 상황 표시 꺼짐)
        mp4_path = "download_video/Feet.mp4"

        if not os.path.exists(mp4_path):
            print(f"테스트 파일이 존재하지 않습니다: {mp4_path}")
            return False

        print("진행 상황 표시: 비활성화 (빠른 테스트)")
        result_text = transcribe_audio_file(mp4_path, language=None, show_progress=False)

        if result_text and len(result_text.strip()) > 0:
            print("진행 상황 표시 비활성화 모드 작동 확인")
            return True
        else:
            print("변환 결과가 비어있음")
            return False

    except Exception as e:
        print(f"진행 상황 표시 테스트 실패: {e}")
        return False


def test_whisper_timestamp_generation():
    """타임스탬프 생성 기능 테스트"""
    print("타임스탬프 생성 기능 테스트 시작...")
    try:
        mp4_path = "download_video/Feet.mp4"

        if not os.path.exists(mp4_path):
            print(f"테스트 파일이 존재하지 않습니다: {mp4_path}")
            return False

        print("타임스탬프 생성 테스트 실행 중...")
        segments = transcribe_audio_with_timestamps(
            mp4_path,
            language=None,
            show_progress=False,
        )

        if not segments:
            print("세그먼트 결과가 비어 있습니다.")
            return False

        first_segment = segments[0]
        required_keys = [
            "index",
            "start",
            "end",
            "start_timecode",
            "end_timecode",
            "text",
        ]

        for key in required_keys:
            if key not in first_segment:
                print(f"세그먼트에 필요한 키가 없습니다: {key}")
                return False

        print("타임스탬프 예시:")
        print(
            f"[{first_segment['start_timecode']} --> {first_segment['end_timecode']}] "
            f"{first_segment['text'][:80]}...",
        )
        return True

    except Exception as e:
        print(f"타임스탬프 테스트 실패: {e}")
        return False


def run_whisper_tests():
    """모든 Whisper 관련 테스트 실행"""
    print("=" * 60)
    print("WHISPER 모듈 테스트 시작")
    print("=" * 60)

    tests = [
        ("모델 로딩", test_whisper_model_loading),
        ("싱글톤 패턴", test_whisper_singleton),
        ("파일 존재하지 않을 때 에러 처리", test_whisper_file_not_found),
        ("MP4 파일 실제 변환 (언어 자동 감지 + 진행 표시)", test_whisper_with_mp4_file),
        ("진행 상황 표시 기능", test_whisper_progress_display),
        ("타임스탬프 생성 기능", test_whisper_timestamp_generation),
        ("유틸리티 메소드", test_whisper_util_methods),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name} 테스트:")
        if test_func():
            passed += 1
        print()

    print("=" * 60)
    print(f"테스트 결과: {passed}/{total} 통과")
    if passed == total:
        print("모든 테스트 통과!")
    else:
        print("일부 테스트 실패")
    print("=" * 60)

    return passed == total


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    # 명령어가 지정되지 않은 경우 도움말 표시
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "youtube":
            analyze_snapshots(
                youtube_url=args.youtube_url,
                count=args.count,
                prompt=args.prompt,
                keep=args.keep,
            )
        elif args.command == "whisper":
            success = run_whisper_tests()
            return 0 if success else 1
        else:
            print(f"알 수 없는 명령어: {args.command}", file=sys.stderr)
            return 1

        return 0
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"오류 발생: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
