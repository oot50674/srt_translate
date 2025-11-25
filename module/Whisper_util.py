"""
Whisper 음성 인식 유틸리티 모듈

CUDA 지원 여부에 따라 GPU large-v3-turbo 또는 CPU tiny 모델을 사용하여
음성 파일을 텍스트로 변환합니다. 진행 상황 표시 기능을 지원합니다.
"""

import os
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

try:  # whisper가 설치되지 않은 환경에서도 모듈 임포트가 가능하도록 처리
    import whisper  # type: ignore
except ImportError:  # pragma: no cover - 런타임 안내 목적
    whisper = None  # type: ignore


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
    PyTorch CUDA 런타임 사용 가능 여부를 확인합니다.
    GPU 지원 모델을 사용하려면 torch가 CUDA를 지원해야 하므로 torch 상태만 신뢰합니다.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class WhisperUtil:
    """
    CUDA 사용 가능 시 GPU 대형 모델, 그렇지 않으면 CPU 경량 모델을 활용하는 유틸리티.
    """

    def __init__(self):
        """
        WhisperUtil 초기화.
        CUDA 사용 가능 시 GPU large-v3-turbo, 아니면 CPU tiny 모델을 로드합니다.
        또한 전사(transcription)에 사용할 기본 파라미터를 설정합니다.
        """
        self.device, self.model_name = self._decide_runtime_mode()
        # 기본 전사 파라미터 (사용자 요청값)
        # TODO: 필요 시 인스턴스 생성 시 외부에서 주입할 수 있도록 확장 가능
        self.no_speech_threshold = 0.8
        self.temperature = 0.6
        self.compression_ratio_threshold = 0.6
        self.model = None
        self._load_model()

    @staticmethod
    def _decide_runtime_mode() -> Tuple[str, str]:
        if _system_has_cuda_runtime():
            return "cuda", "large-v3-turbo"
        return "cpu", "tiny"

    def _load_model(self):
        """
        환경에 맞는 Whisper 모델을 로드합니다.
        """
        mode_label = (
            "GPU Whisper large-v3-turbo 모델"
            if self.device == "cuda"
            else "CPU Whisper tiny 모델"
        )
        try:
            if whisper is None:
                raise ImportError("openai-whisper 패키지가 설치되어 있지 않습니다.")
            print(f"{mode_label} 로딩 중...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print("모델 로딩 완료")
        except Exception as e:
            if self.device == "cuda" and whisper is not None:
                print(f"GPU 로딩 실패({e}). CPU tiny 모델로 폴백합니다.")
                self.device = "cpu"
                self.model_name = "tiny"
                try:
                    print("CPU Whisper tiny 모델 로딩 중...")
                    self.model = whisper.load_model(self.model_name, device=self.device)
                    print("모델 로딩 완료 (CPU 폴백)")
                    return
                except Exception as fallback_exc:
                    print(f"CPU 폴백 로딩 실패: {fallback_exc}")
                    raise
            print(f"모델 로딩 실패: {e}")
            raise

    def _ensure_model_loaded(self) -> None:
        """
        현재 모델이 로드되어 있지 않으면 새로 로드합니다.
        """
        if self.model is None:
            self._load_model()

    def unload_model(self) -> None:
        """
        로드된 Whisper 모델을 언로드하여 GPU/메모리를 반환합니다.
        """
        if self.model is None:
            return
        try:
            import torch  # type: ignore
        except ImportError:
            torch = None
        self.model = None
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print("Whisper 모델을 언로드했습니다.")

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        show_progress: bool = True,
        no_speech_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
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
        if self.model is None:
            raise Exception("모델이 로드되지 않았습니다.")

        try:
            print(f"파일 변환 중: {audio_path}")
            if language:
                print(f"지정된 언어: {language}")
            else:
                print("언어 자동 감지 모드")

            # 진행 상황 표시 설정
            verbose = show_progress
            if show_progress:
                print("변환 시작... (진행 상황이 표시됩니다)")
                print("-" * 50)

            start_time = time.time()

            # 전사 파라미터: 메서드 호출 시 전달된 값이 우선, 없으면 인스턴스 기본값 사용
            no_speech_threshold = self.no_speech_threshold if no_speech_threshold is None else no_speech_threshold
            temperature = self.temperature if temperature is None else temperature
            compression_ratio_threshold = (
                self.compression_ratio_threshold
                if compression_ratio_threshold is None
                else compression_ratio_threshold
            )
            # whisper.transcribe에 전달할 인자 구성
            transcribe_kwargs: Dict[str, Any] = {}
            if language:
                transcribe_kwargs["language"] = language
            # no_speech_threshold, compression_ratio_threshold, chunk_length_s는 float
            transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
            transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
            # temperature는 float 또는 리스트로 허용하므로 단일 값이면 리스트로 감싸는 것이 안전
            transcribe_kwargs["temperature"] = [temperature] if isinstance(temperature, (int, float)) else temperature
            transcribe_kwargs["verbose"] = verbose

            # 변환 실행
            result = self.model.transcribe(audio_path, **transcribe_kwargs)

            end_time = time.time()
            duration = end_time - start_time

            if show_progress:
                print("-" * 50)

            # 감지된 언어 정보 출력
            detected_lang = result.get("language", "알 수 없음")
            print(f"감지된 언어: {detected_lang}")
            print(f"소요 시간: {duration:.1f}초")
            print("변환 완료")
            return result

        except Exception as e:
            print(f"변환 실패: {e}")
            raise

    def get_text_only(self, audio_path: str, language: Optional[str] = None, show_progress: bool = True) -> str:
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
