"""
Whisper GPU 진단용 스크립트.

예제 실행:
    python whisper_gpu_diag.py --audio test.mp3 --model large-v3 --compute int8_float16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable


def _print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def _print_env(keys: Iterable[str]) -> None:
    for key in keys:
        value = os.environ.get(key)
        print(f"{key}: {value if value is not None else ''}")


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except FileNotFoundError:
        return "not found"
    except subprocess.CalledProcessError as exc:
        return f"failed (code {exc.returncode}):\n{exc.output}"


def _check_nvidia() -> None:
    _print_header("nvidia-smi -L")
    print(_run_cmd(["nvidia-smi", "-L"]))

    _print_header("nvidia-smi -q -d MEMORY")
    print(_run_cmd(["nvidia-smi", "-q", "-d", "MEMORY"]))


def _check_torch() -> None:
    _print_header("torch.cuda 상태")
    try:
        import torch  # type: ignore

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                print(f"device_name[0]: {torch.cuda.get_device_name(0)}")
            except Exception as exc:  # pragma: no cover - best effort
                print(f"device_name 조회 실패: {exc}")
    except Exception as exc:
        print(f"torch import 실패: {exc}")


def _check_ctranslate2() -> None:
    _print_header("ctranslate2 상태")
    try:
        import ctranslate2  # type: ignore

        print(f"ctranslate2: {getattr(ctranslate2, '__version__', 'unknown')}")
        if hasattr(ctranslate2, "get_cuda_device_count"):
            try:
                print(f"cuda_device_count: {ctranslate2.get_cuda_device_count()}")
            except Exception as exc:
                print(f"get_cuda_device_count 실패: {exc}")
        if hasattr(ctranslate2, "has_cuda"):
            try:
                print(f"has_cuda: {bool(ctranslate2.has_cuda())}")
            except Exception as exc:
                print(f"has_cuda 실패: {exc}")
    except Exception as exc:
        print(f"ctranslate2 import 실패: {exc}")


def _run_whisper(audio_path: str, model: str, compute: str) -> None:
    _print_header("faster-whisper 로드 및 전사")
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        print(f"faster-whisper import 실패: {exc}")
        return

    try:
        print(f"모델 로드: {model} on cuda ({compute})")
        model_inst = WhisperModel(
            model,
            device="cuda",
            compute_type=compute,
            local_files_only=False,
        )
    except Exception as exc:
        print(f"모델 로드 실패: {exc}")
        return

    try:
        segments, info = model_inst.transcribe(
            audio_path,
            beam_size=1,
            vad_filter=False,
            language=None,
        )
        print(f"감지 언어: {getattr(info, 'language', None)}")
        print(f"언어 확률: {getattr(info, 'language_probability', None)}")
        print("첫 3개 세그먼트:")
        for idx, seg in enumerate(segments):
            if idx >= 3:
                break
            print(f"- {seg.start:.2f}~{seg.end:.2f}s: {seg.text}")
    except Exception as exc:
        print(f"전사 실패: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper GPU 진단 스크립트")
    parser.add_argument("--audio", default="test.mp3", help="테스트 오디오 파일 경로")
    parser.add_argument("--model", default="large-v3", help="Whisper 모델 크기")
    parser.add_argument(
        "--compute",
        default="int8_float16",
        help="compute_type (예: int8_float16, float16, int8)",
    )
    args = parser.parse_args()

    _print_header("환경 변수")
    _print_env(
        [
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "PATH",
        ]
    )

    _check_nvidia()
    _check_torch()
    _check_ctranslate2()
    _run_whisper(args.audio, args.model, args.compute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
