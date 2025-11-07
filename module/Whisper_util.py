"""
Whisper 음성 인식 유틸리티 모듈

CPU 전용으로 Whisper tiny 모델을 사용하여 음성 파일을 텍스트로 변환합니다.
진행 상황 표시 기능을 지원합니다.
"""

import whisper
import torch
import os
from typing import Optional, Dict, Any, Callable, List
import time


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


class WhisperUtil:
    """
    CPU 전용 Whisper 음성 인식 유틸리티 클래스
    """

    def __init__(self):
        """
        WhisperUtil 초기화
        CPU 전용으로 tiny 모델을 로드합니다.
        """
        self.device = "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        CPU 전용 tiny 모델을 로드합니다.
        """
        try:
            print("CPU 전용 Whisper tiny 모델 로딩 중...")
            self.model = whisper.load_model("tiny", device=self.device)
            print("모델 로딩 완료 (CPU 전용)")
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            raise

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None, show_progress: bool = True) -> Dict[str, Any]:
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

            # 변환 실행
            if language:
                result = self.model.transcribe(audio_path, language=language, verbose=verbose)
            else:
                result = self.model.transcribe(audio_path, verbose=verbose)

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


def get_whisper_util() -> WhisperUtil:
    """
    WhisperUtil 싱글톤 인스턴스를 반환합니다.

    Returns:
        WhisperUtil: Whisper 유틸리티 인스턴스
    """
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperUtil()
    return _whisper_instance


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
