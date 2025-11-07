import os
import torch
import torchaudio
import numpy as np
import sounddevice as sd
from typing import List, Dict, Optional, Tuple, Any


class SileroVAD:
    """
    Silero VAD (Voice Activity Detection) 모듈
    CPU 전용으로 구현되어 CUDA 없이 사용할 수 있습니다.
    """

    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.5,
                 min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30, num_threads: int = 1):
        """
        Silero VAD 초기화

        Args:
            sampling_rate: 샘플링 레이트 (16000 또는 8000 권장)
            threshold: 음성 감지 임계값 (0.35~0.7, 낮을수록 민감)
            min_speech_duration_ms: 최소 음성 지속 시간 (ms)
            min_silence_duration_ms: 최소 침묵 지속 시간 (ms)
            speech_pad_ms: 음성 구간 앞뒤 여유 시간 (ms)
            num_threads: PyTorch 스레드 수 (1~2 권장)
        """
        # CUDA 비활성화 및 CPU 전용 설정
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(num_threads)

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.num_threads = num_threads

        # 모델 및 유틸리티 로드
        self.model = None
        self.utils = None
        self._load_model()

    def _load_model(self):
        """Silero VAD 모델 로드"""
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False
            )
            # CPU로 명시적 이동
            self.model = self.model.cpu()
            print("Silero VAD 모델 로드 완료 (CPU 모드)")
        except Exception as e:
            raise RuntimeError(f"Silero VAD 모델 로드 실패: {e}")

    def get_utils(self):
        """VAD 유틸리티 함수들 반환"""
        if self.utils is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        return self.utils

    def detect_speech_from_file(self, audio_path: str) -> List[Dict[str, float]]:
        """
        오디오 파일에서 음성 구간 검출

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            음성 구간 리스트 [{'start': 초, 'end': 초}, ...]
        """
        if self.utils is None:
            raise RuntimeError("모델이 로드되지 않았습니다")

        get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.utils

        # 오디오 파일 읽기 (내부에서 모노/정규화 처리)
        try:
            wav = read_audio(audio_path, sampling_rate=self.sampling_rate)
        except Exception as e:
            raise RuntimeError(f"오디오 파일 읽기 실패: {e}")

        # 음성 구간 검출
        speech_timestamps = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            return_seconds=True,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms
        )

        return speech_timestamps

    def extract_speech_audio(self, audio_path: str, output_path: str) -> bool:
        """
        오디오 파일에서 음성 구간만 추출하여 저장

        Args:
            audio_path: 입력 오디오 파일 경로
            output_path: 출력 오디오 파일 경로

        Returns:
            성공 여부
        """
        if self.utils is None:
            raise RuntimeError("모델이 로드되지 않았습니다")

        get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.utils

        try:
            # 오디오 파일 읽기
            wav = read_audio(audio_path, sampling_rate=self.sampling_rate)

            # 음성 구간 검출
            speech_timestamps = get_speech_timestamps(
                wav,
                self.model,
                sampling_rate=self.sampling_rate,
                return_seconds=True,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms
            )

            # 음성 구간만 모아서 저장
            if speech_timestamps:
                speech_audio = collect_chunks(speech_timestamps, wav)
                save_audio(output_path, speech_audio, sampling_rate=self.sampling_rate)
                return True
            else:
                print("음성 구간이 검출되지 않았습니다")
                return False

        except Exception as e:
            print(f"음성 추출 실패: {e}")
            return False

    def resample_audio(self, audio_path: str, target_sr: Optional[int] = None) -> torch.Tensor:
        """
        오디오 파일을 지정된 샘플레이트로 리샘플링하고 모노로 변환

        Args:
            audio_path: 오디오 파일 경로
            target_sr: 목표 샘플레이트 (None이면 self.sampling_rate 사용)

        Returns:
            리샘플링된 오디오 텐서 [samples]
        """
        if target_sr is None:
            target_sr = self.sampling_rate

        try:
            # 오디오 로드
            wav, sr = torchaudio.load(audio_path)  # [channels, samples]

            # 리샘플링
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            # 모노화 (여러 채널 평균)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 1D 텐서로 변환
            wav = wav.squeeze(0)  # [samples]

            return wav

        except Exception as e:
            raise RuntimeError(f"오디오 리샘플링 실패: {e}")

    def start_streaming_vad(self, callback: callable = None, blocksize: int = 512) -> None:
        """
        마이크 스트리밍으로 실시간 VAD 수행

        Args:
            callback: 음성 감지 시 호출될 콜백 함수
                     함수 시그니처: callback(speech_info: List[Dict])
            blocksize: 오디오 블록 크기 (512=16kHz에서 32ms)
        """
        if self.utils is None:
            raise RuntimeError("모델이 로드되지 않았습니다")

        get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.utils

        # VAD 이터레이터 생성
        vad_iterator = VADIterator(self.model, sampling_rate=self.sampling_rate)

        def audio_callback(indata, frames, time, status):
            """오디오 콜백 함수"""
            if status:
                print(f"오디오 상태: {status}")

            # 모노 float32로 변환
            mono = indata[:, 0].copy()
            audio_tensor = torch.from_numpy(mono).float()

            # 음성 구간 감지
            speech_info = vad_iterator(audio_tensor, return_seconds=True)

            # 음성이 감지되면 콜백 호출
            if speech_info and callback:
                callback(speech_info)

        print(f"마이크 스트리밍 시작... (샘플레이트: {self.sampling_rate}Hz, 블록 크기: {blocksize})")
        print("Ctrl+C로 중지")

        try:
            # 입력 스트림 시작
            with sd.InputStream(
                channels=1,
                samplerate=self.sampling_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=audio_callback
            ):
                while True:
                    sd.sleep(1000)  # 1초 대기

        except KeyboardInterrupt:
            print("스트리밍 중지")
        except Exception as e:
            print(f"스트리밍 오류: {e}")
        finally:
            vad_iterator.reset_states()

    def update_parameters(self, **kwargs):
        """
        VAD 매개변수 업데이트

        Args:
            threshold: 음성 감지 임계값
            min_speech_duration_ms: 최소 음성 지속 시간
            min_silence_duration_ms: 최소 침묵 지속 시간
            speech_pad_ms: 음성 패딩 시간
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"{key}: {value}")

    def get_parameters(self) -> Dict[str, Any]:
        """현재 VAD 매개변수 반환"""
        return {
            'sampling_rate': self.sampling_rate,
            'threshold': self.threshold,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'speech_pad_ms': self.speech_pad_ms,
            'num_threads': self.num_threads
        }


# 편의를 위한 함수들
def detect_speech_from_file(audio_path: str, **vad_params) -> List[Dict[str, float]]:
    """
    간단한 파일 음성 검출 함수

    Args:
        audio_path: 오디오 파일 경로
        **vad_params: VAD 매개변수들

    Returns:
        음성 구간 리스트
    """
    vad = SileroVAD(**vad_params)
    return vad.detect_speech_from_file(audio_path)


def extract_speech_audio(input_path: str, output_path: str, **vad_params) -> bool:
    """
    간단한 음성 추출 함수

    Args:
        input_path: 입력 오디오 파일 경로
        output_path: 출력 오디오 파일 경로
        **vad_params: VAD 매개변수들

    Returns:
        성공 여부
    """
    vad = SileroVAD(**vad_params)
    return vad.extract_speech_audio(input_path, output_path)


def resample_audio_to_mono(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    오디오를 모노로 리샘플링하는 유틸리티 함수

    Args:
        audio_path: 오디오 파일 경로
        target_sr: 목표 샘플레이트

    Returns:
        리샘플링된 오디오 텐서
    """
    vad = SileroVAD(sampling_rate=target_sr)
    return vad.resample_audio(audio_path, target_sr)


if __name__ == "__main__":
    # 사용 예제
    print("Silero VAD 모듈 테스트")

    # 기본 VAD 인스턴스 생성
    vad = SileroVAD(
        sampling_rate=16000,
        threshold=0.5,
        num_threads=1
    )

    print("매개변수:", vad.get_parameters())

    # 파일이 있다면 테스트
    import os
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"\n{test_file}에서 음성 검출 테스트...")
        speech_segments = vad.detect_speech_from_file(test_file)
        print(f"감지된 음성 구간: {speech_segments}")
    else:
        print(f"\n테스트 파일 {test_file}이 없습니다.")
