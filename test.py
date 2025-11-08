import argparse
import time
from typing import Optional

import torch

try:
    import whisper
except ImportError as whisper_import_error:  # pragma: no cover
    whisper = None  # type: ignore


def print_torch_environment() -> None:
    """파이토치 및 CUDA 환경 정보를 출력한다."""
    print("=== PyTorch / CUDA 환경 점검 ===")
    print(f"torch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"CUDA 장치 수: {device_count}")
    if device_count > 0:
        for index in range(device_count):
            device_name = torch.cuda.get_device_name(index)
            compute_capability = torch.cuda.get_device_capability(index)
            print(f" - [{index}] {device_name} (연산 능력: {compute_capability})")


def try_whisper_load(
    model_name: str,
    device: str,
    download_root: Optional[str],
    fallback_model: str,
) -> bool:
    """Whisper 모델을 지정한 장치에 로딩해보고 실패 시 CPU 폴백을 시도한다."""
    if whisper is None:
        print("whisper 모듈을 찾을 수 없습니다. 먼저 `pip install -r requirements.txt`를 수행하세요.")
        return False

    print(f"\n=== Whisper 모델 로딩 테스트 ({model_name}, device={device}) ===")
    start = time.perf_counter()
    try:
        whisper.load_model(model_name, device=device, download_root=download_root)
    except RuntimeError as runtime_error:
        duration = time.perf_counter() - start
        print(f"GPU 로딩 실패 ({duration:.2f}s): {runtime_error}")
        if device == "cuda":
            print("CUDA 로딩 실패로 CPU 폴백을 시도합니다.")
            return try_cpu_fallback(model_name, download_root, fallback_model)
        return False
    except Exception as unexpected_error:  # pragma: no cover
        duration = time.perf_counter() - start
        print(f"예상치 못한 예외 발생 ({duration:.2f}s): {unexpected_error}")
        if device == "cuda":
            print("CUDA 로딩 실패로 CPU 폴백을 시도합니다.")
            return try_cpu_fallback(model_name, download_root, fallback_model)
        return False
    else:
        duration = time.perf_counter() - start
        print(f"Whisper 모델 로딩 성공 ({duration:.2f}s)")
        return True


def try_cpu_fallback(model_name: str, download_root: Optional[str], fallback_model: str) -> bool:
    """CPU 환경에서 원본 또는 폴백 모델 로딩을 시도한다."""
    start = time.perf_counter()
    try:
        whisper.load_model(model_name, device="cpu", download_root=download_root)
        duration = time.perf_counter() - start
        print(f"CPU 로딩 성공 ({duration:.2f}s) - 동일 모델 유지")
        return True
    except RuntimeError as cpu_error:
        duration = time.perf_counter() - start
        print(f"CPU 동일 모델 로딩 실패 ({duration:.2f}s): {cpu_error}")
        if fallback_model == model_name:
            return False
        print(f"폴백 모델({fallback_model})로 재시도합니다.")
        return try_cpu_fallback(fallback_model, download_root, fallback_model)
    except Exception as unexpected_error:  # pragma: no cover
        duration = time.perf_counter() - start
        print(f"CPU 폴백 중 예외 발생 ({duration:.2f}s): {unexpected_error}")
        return False


def parse_arguments() -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="CUDA 환경과 Whisper 모델 로딩을 점검하는 스크립트",
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="테스트할 Whisper 모델 이름 (기본값: large-v3-turbo)",
    )
    parser.add_argument(
        "--fallback-model",
        default="tiny",
        help="GPU 로딩 실패 시 CPU 폴백으로 시도할 Whisper 모델 (기본값: tiny)",
    )
    parser.add_argument(
        "--download-root",
        default=None,
        help="모델 다운로드 캐시 경로 지정 (기본값: whisper 기본 경로)",
    )
    parser.add_argument(
        "--force-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="모델 로딩에 사용할 장치 강제 지정 (기본값: auto)",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Whisper 모델 로딩 테스트를 건너뜁니다.",
    )
    return parser.parse_args()


def resolve_device(force_device: str) -> str:
    """사용자 지정에 따라 모델 로딩 장치를 결정한다."""
    if force_device == "cpu":
        return "cpu"
    if force_device == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    """쿠다 환경과 Whisper 모델 로딩을 테스트한다."""
    args = parse_arguments()
    print_torch_environment()
    device = resolve_device(args.force_device)
    print(f"\n선택된 모델 로딩 장치: {device}")
    if args.skip_whisper:
        print("Whisper 로딩 테스트는 건너뜁니다.")
        return
    try_whisper_load(args.model, device, args.download_root, args.fallback_model)


if __name__ == "__main__":
    main()

