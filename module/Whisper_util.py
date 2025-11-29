"""
Whisper 음성 인식 유틸리티 모듈

faster-whisper 기반 WhisperEngine을 사용해 음성 파일을 텍스트로 변환합니다.
CUDA 지원 여부에 따라 GPU large-v3 또는 CPU int8 모델을 선택하며, 진행 상황 표시 기능을 지원합니다.
"""

import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from module.WhisperEngine import (
    DEFAULT_MODEL_PARAMS,
    DEFAULT_TRANSCRIBE_PARAMS,
    WhisperEngine,
)
from module.whisper_process import WhisperEngineInitError, WhisperProcessRunner


def _format_timestamp(seconds: float) -> str:
    """
    초 단위 시간을 SRT 형식의 타임스탬프로 변환합니다.

    Args:
        seconds (float): 변환할 초 단위 시간

    Returns:
        str: HH:MM:SS,mmm 형식의 문자열
    """
    try:
        total_millis = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        total_millis = 0

    if total_millis < 0:
        total_millis = 0

    hours = total_millis // 3600000
    minutes = (total_millis % 3600000) // 60000
    secs = (total_millis % 60000) // 1000
    millis = total_millis % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _system_has_cuda_runtime() -> bool:
    """
    CUDA 사용 가능 여부를 확인합니다.
    torch가 CPU 휠이어도 ctranslate2가 CUDA를 지원할 수 있으므로 둘 다 확인합니다.
    """
    # 1) torch를 통한 확인 (있다면)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True
    except Exception:
        pass

    # 2) ctranslate2 CUDA 지원 여부 확인
    try:
        import ctranslate2  # type: ignore

        if hasattr(ctranslate2, "get_cuda_device_count"):
            try:
                if int(ctranslate2.get_cuda_device_count()) > 0:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


class WhisperUtil:
    """
    CUDA 사용 가능 시 GPU 모델, 그렇지 않으면 CPU 양자화 모델을 활용하는 유틸리티.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        **model_kwargs: Any,
    ):
        """
        WhisperUtil 초기화.
        CUDA 사용 가능 시 GPU large-v3, 아니면 CPU int8 모델을 로드합니다.
        또한 전사(transcription)에 사용할 기본 파라미터를 설정합니다.
        """
        self.device, self.compute_type = self._decide_runtime_mode(device, compute_type)
        self.model_name = model_name or "large-v3"
        self.download_root = download_root
        self.model_kwargs = model_kwargs
        self.no_speech_threshold = 0.7
        self.temperature = 0.2
        self.compression_ratio_threshold = 2.2
        self.repetition_penalty = 1.1
        self.no_repeat_ngram_size = 3
        self.engine: Optional[WhisperEngine] = None  # 타입 힌트 유지
        self.process_runner: Optional[WhisperProcessRunner] = None
        self._init_process_runner()

    @staticmethod
    def _decide_runtime_mode(
        device: Optional[str], compute_type: Optional[str]
    ) -> Tuple[str, str]:
        env_device = (os.environ.get("WHISPER_DEVICE") or "").lower().strip()
        env_compute = (os.environ.get("WHISPER_COMPUTE_TYPE") or "").lower().strip()

        if env_device in {"cuda", "cpu"}:
            chosen_device = env_device
        elif device:
            chosen_device = device
        else:
            chosen_device = "cuda" if _system_has_cuda_runtime() else "cpu"

        chosen_compute = compute_type or env_compute
        if not chosen_compute:
            if chosen_device == "cuda":
                chosen_compute = "int8_float16"
            else:
                chosen_compute = DEFAULT_MODEL_PARAMS.get("compute_type", "int8")

        return chosen_device, chosen_compute

    def _load_model(self) -> None:
        """
        환경에 맞는 Whisper 모델을 로드합니다.
        """
        mode_label = (
            f"GPU faster-whisper {self.model_name} 모델"
            if self.device == "cuda"
            else f"CPU faster-whisper {self.model_name} 모델"
        )
        print(f"{mode_label} 프로세스 초기화 준비 완료.")

    def _ensure_model_loaded(self) -> None:
        """
        현재 모델이 로드되어 있지 않으면 새로 로드합니다.
        """
        if self.process_runner is None:
            self._init_process_runner()

    def _init_process_runner(self) -> None:
        """
        프로세스 런너를 초기화합니다.
        """
        self.process_runner = WhisperProcessRunner(
            model_name=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
            **self.model_kwargs,
        )
        self._load_model()

    def _restart_process_runner(self, device: str, compute_type: str) -> None:
        """
        런너를 지정된 장치/정밀도로 재시작합니다.
        """
        if self.process_runner:
            try:
                self.process_runner.stop()
            except Exception:
                pass
        self.device = device
        self.compute_type = compute_type
        self.process_runner = None
        self._init_process_runner()

    def unload_model(self) -> None:
        """
        로드된 Whisper 모델을 언로드하여 GPU/메모리를 반환합니다.
        """
        if self.process_runner is None:
            return
        try:
            self.process_runner.stop()
        except Exception:
            pass
        self.process_runner = None
        print("Whisper 모델 프로세스를 종료했습니다.")

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        show_progress: bool = True,
        no_speech_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        음성/비디오 파일을 텍스트로 변환합니다.

        Args:
            audio_path (str): 변환할 음성/비디오 파일 경로
            language (str, optional): 언어 코드 (None이면 자동 감지, "ko"는 한국어)
            show_progress (bool): 진행 상황 표시 여부 (기본값: True)

        Returns:
            Dict[str, Any]: 변환 결과 (text, segments 등 포함)

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            Exception: 변환 중 오류 발생 시
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

        self._ensure_model_loaded()
        if self.process_runner is None:
            raise Exception("모델 프로세스가 초기화되지 않았습니다.")

        try:
            print(f"파일 변환 중: {audio_path}")
            if language:
                print(f"지정된 언어: {language}")
            else:
                print("언어 자동 감지 모드")

            if show_progress:
                print("변환 시작... (진행 상황이 표시됩니다)")
                print("-" * 50)

            start_time = time.time()

            # 전사 파라미터: 메서드 호출 시 전달된 값이 우선, 없으면 인스턴스 기본값 사용
            no_speech_value = (
                self.no_speech_threshold
                if no_speech_threshold is None
                else no_speech_threshold
            )
            temperature_value = (
                self.temperature if temperature is None else temperature
            )
            compression_value = (
                self.compression_ratio_threshold
                if compression_ratio_threshold is None
                else compression_ratio_threshold
            )
            repetition_penalty_value = (
                self.repetition_penalty if repetition_penalty is None else repetition_penalty
            )
            no_repeat_value = (
                self.no_repeat_ngram_size if no_repeat_ngram_size is None else no_repeat_ngram_size
            )
            transcribe_kwargs: Dict[str, Any] = dict(DEFAULT_TRANSCRIBE_PARAMS)
            if language:
                transcribe_kwargs["language"] = language
            transcribe_kwargs["no_speech_threshold"] = no_speech_value
            transcribe_kwargs["compression_ratio_threshold"] = compression_value
            transcribe_kwargs["temperature"] = temperature_value
            transcribe_kwargs["repetition_penalty"] = repetition_penalty_value
            transcribe_kwargs["no_repeat_ngram_size"] = no_repeat_value

            try:
                result = self.process_runner.transcribe(
                    audio_path,
                    progress_callback=progress_callback,
                    **transcribe_kwargs,
                )
            except WhisperEngineInitError as exc:
                # CPU 폴백을 배제하고 GPU에서만 동작하도록 강제
                raise RuntimeError(
                    f"GPU Whisper 모델 초기화 실패: {exc}"
                ) from exc
            segments_list = result.get("segments", [])
            collected_texts: List[str] = []
            for segment in segments_list:
                text_value = str(segment.get("text", "")).strip()
                if text_value:
                    collected_texts.append(text_value)

            end_time = time.time()
            duration = end_time - start_time

            if show_progress:
                print("-" * 50)

            info = result.get("info") or {}
            detected_lang = info.get("language") or language or "unknown"
            language_probability = info.get("language_probability")
            if detected_lang:
                print(f"감지된 언어: {detected_lang}")
            print(f"소요 시간: {duration:.1f}초")
            print("변환 완료")

            return {
                "text": result.get("text") or " ".join(collected_texts).strip(),
                "segments": segments_list,
                "language": detected_lang,
                "language_probability": language_probability,
            }

        except Exception as e:
            print(f"변환 실패: {e}")
            raise

    def get_text_only(
        self, audio_path: str, language: Optional[str] = None, show_progress: bool = True
    ) -> str:
        """
        음성/비디오 파일에서 텍스트만 추출합니다.

        Args:
            audio_path (str): 변환할 음성/비디오 파일 경로
            language (str, optional): 언어 코드 (None이면 자동 감지, "ko"는 한국어)
            show_progress (bool): 진행 상황 표시 여부 (기본값: True)

        Returns:
            str: 변환된 텍스트
        """
        result = self.transcribe_audio(audio_path, language, show_progress)
        return result.get("text", "").strip()

    def get_segments_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        음성/비디오 파일에서 세그먼트별 텍스트와 타임스탬프를 추출합니다.

        Args:
            audio_path (str): 변환할 음성/비디오 파일 경로
            language (str, optional): 언어 코드 (None이면 자동 감지)
            show_progress (bool): 진행 상황 표시 여부

        Returns:
            List[Dict[str, Any]]: 세그먼트별 텍스트와 타임스탬프 정보
        """
        result = self.transcribe_audio(audio_path, language, show_progress)
        segments = result.get("segments", [])

        formatted_segments: List[Dict[str, Any]] = []
        for index, segment in enumerate(segments, start=1):
            start_value = segment.get("start", 0.0)
            end_value = segment.get("end", 0.0)
            text_value = segment.get("text", "")

            start_seconds = float(start_value) if start_value is not None else 0.0
            end_seconds = float(end_value) if end_value is not None else 0.0

            formatted_segments.append(
                {
                    "index": index,
                    "start": start_seconds,
                    "end": end_seconds,
                    "start_timecode": _format_timestamp(start_seconds),
                    "end_timecode": _format_timestamp(end_seconds),
                    "text": text_value.strip(),
                }
            )

        return formatted_segments


# 전역 인스턴스 (싱글톤 패턴)
_whisper_instance: Optional[WhisperUtil] = None
_whisper_lock = threading.Lock()


def get_whisper_util() -> WhisperUtil:
    """
    WhisperUtil 싱글톤 인스턴스를 반환합니다.

    Returns:
        WhisperUtil: Whisper 유틸리티 인스턴스
    """
    global _whisper_instance
    with _whisper_lock:
        if _whisper_instance is None:
            _whisper_instance = WhisperUtil()
        return _whisper_instance


def unload_whisper_util() -> None:
    """
    글로벌 Whisper 인스턴스를 해제합니다.
    """
    global _whisper_instance
    with _whisper_lock:
        if _whisper_instance is None:
            return
        try:
            _whisper_instance.unload_model()
        except Exception:
            pass
        _whisper_instance = None


def transcribe_audio_file(audio_path: str, language: Optional[str] = None, show_progress: bool = True) -> str:
    """
    간단한 인터페이스로 음성/비디오 파일을 텍스트로 변환합니다.

    Args:
        audio_path (str): 변환할 음성/비디오 파일 경로
        language (str, optional): 언어 코드 (None이면 자동 감지, "ko"는 한국어)
        show_progress (bool): 진행 상황 표시 여부 (기본값: True)

    Returns:
        str: 변환된 텍스트
    """
    util = get_whisper_util()
    return util.get_text_only(audio_path, language, show_progress)


def transcribe_audio_with_timestamps(
    audio_path: str,
    language: Optional[str] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    음성/비디오 파일을 세그먼트 단위로 변환하고 타임스탬프 정보를 제공합니다.

    Args:
        audio_path (str): 변환할 음성/비디오 파일 경로
        language (str, optional): 언어 코드 (None이면 자동 감지)
        show_progress (bool): 진행 상황 표시 여부

    Returns:
        List[Dict[str, Any]]: 세그먼트별 텍스트와 타임스탬프 정보
    """
    util = get_whisper_util()
    return util.get_segments_with_timestamps(audio_path, language, show_progress)
