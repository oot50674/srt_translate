"""Silero VAD 래퍼 클래스를 별도 파일로 분리합니다."""
from __future__ import annotations

from typing import Dict, List
import os
import tempfile
import subprocess
import logging

import torch

logger = logging.getLogger(__name__)


def _extract_audio_to_wav(input_path: str, output_path: str) -> None:
    """비디오 파일에서 오디오를 WAV 형식으로 추출합니다.

    Args:
        input_path: 입력 비디오/오디오 파일 경로
        output_path: 출력 WAV 파일 경로

    Raises:
        RuntimeError: FFmpeg 실행 실패 시
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vn",  # 비디오 스트림 제외
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",  # 16kHz 샘플링 레이트 (Silero VAD 기본값)
        "-ac", "1",  # 모노
        "-y",  # 덮어쓰기
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.debug(f"오디오 추출 완료: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"오디오 추출 실패: {error_msg}") from e
    except FileNotFoundError:
        raise RuntimeError("FFmpeg를 찾을 수 없습니다. FFmpeg가 설치되어 있는지 확인하세요.") from None


class SileroVAD:
    """Torch Hub의 Silero 모델을 직접 로드해 음성 구간을 검출하는 간단한 래퍼."""

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.7,
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

        # 비디오 파일 확장자 확인
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
        file_ext = os.path.splitext(audio_path)[1].lower()

        # 비디오 파일인 경우 오디오 추출
        temp_wav_path = None
        if file_ext in video_extensions:
            logger.info(f"비디오 파일 감지됨 ({file_ext}), 오디오 추출 중...")
            temp_wav_path = tempfile.mktemp(suffix='.wav')
            try:
                _extract_audio_to_wav(audio_path, temp_wav_path)
                audio_path_to_read = temp_wav_path
            except Exception as exc:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                raise RuntimeError(f"비디오에서 오디오 추출 실패: {exc}") from exc
        else:
            audio_path_to_read = audio_path

        try:
            wav = read_audio(audio_path_to_read, sampling_rate=self.sampling_rate)
        except Exception as exc:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            raise RuntimeError(f"오디오 파일 읽기 실패: {exc}") from exc
        finally:
            # 임시 WAV 파일 정리
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                    logger.debug(f"임시 파일 삭제: {temp_wav_path}")
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")

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


