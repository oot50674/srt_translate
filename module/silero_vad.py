"""Silero VAD 래퍼 클래스를 별도 파일로 분리합니다."""
from __future__ import annotations

from typing import Dict, List

import torch


class SileroVAD:
    """Torch Hub의 Silero 모델을 직접 로드해 음성 구간을 검출하는 간단한 래퍼."""

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 250,
        speech_pad_ms: int = 30,
        num_threads: int = 1,
    ) -> None:
        torch.set_num_threads(num_threads)

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.num_threads = num_threads

        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False,
            )
            self.model = self.model.cpu()
        except Exception as exc:
            raise RuntimeError(f"Silero VAD 모델 로드 실패: {exc}") from exc

    def detect_speech_from_file(self, audio_path: str) -> List[Dict[str, float]]:
        if self.utils is None or self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = self.utils
        try:
            wav = read_audio(audio_path, sampling_rate=self.sampling_rate)
        except Exception as exc:
            raise RuntimeError(f"오디오 파일 읽기 실패: {exc}") from exc

        return get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=True,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )


