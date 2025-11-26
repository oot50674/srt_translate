import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator

from module.cuda_runtime import MODULE_DIR, register_embedded_cuda

# 윈도우 인코딩 설정 (모듈 로드 시 실행)
sys.stdout.reconfigure(encoding='utf-8')

# Embedded CUDA runtime 경로를 먼저 등록
register_embedded_cuda()

BASE_DIR = str(MODULE_DIR.parent)
MODELS_DIR = os.path.join(BASE_DIR, "models")

from faster_whisper import WhisperModel

# --- [기본 설정값 정의] ---
# 모델 초기화 기본값
DEFAULT_MODEL_PARAMS = {
    "model_size_or_path": "large-v3",
    "device": "cuda",
    "compute_type": "int8",
    "download_root": MODELS_DIR,
    "local_files_only": False,
}

# 전사(Transcribe) 기본값
DEFAULT_TRANSCRIBE_PARAMS = {
    "beam_size": 5,
    "no_speech_threshold": 0.7,
    "temperature": 0.5,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "vad_filter": True,
    "vad_parameters": dict(min_silence_duration_ms=700),
    "condition_on_previous_text": False,
    "word_timestamps": True, # 필요시 True
    "chunk_length": 30,
    # note: faster-whisper transcribe expects 'chunk_length' (seconds)
}

class WhisperEngine:
    def __init__(self, model_name: str = None, device: str = None, **kwargs):
        """
        Whisper 모델을 초기화합니다.
        :param model_name: 모델 크기 (예: 'large-v3', 'base')
        :param device: 실행 장치 ('cuda' or 'cpu')
        :param kwargs: WhisperModel 초기화에 필요한 추가 파라미터들
        """
        # 기본 설정에 사용자 입력 덮어쓰기
        self.model_params = DEFAULT_MODEL_PARAMS.copy()
        if model_name: self.model_params["model_size_or_path"] = model_name
        if device: self.model_params["device"] = device
        
        # download_root가 None으로 전달되면 기본값을 덮어쓰지 않도록 제거
        if "download_root" in kwargs and kwargs["download_root"] is None:
            del kwargs["download_root"]

        # 나머지 kwargs 업데이트
        self.model_params.update(kwargs)

        # 모델 다운로드 경로가 없으면 자동 생성
        download_root = self.model_params.get("download_root")
        if download_root and not os.path.exists(download_root):
            try:
                os.makedirs(download_root, exist_ok=True)
                print(f"[-] Created model directory: {download_root}")
            except Exception as e:
                print(f"[!] Failed to create model directory '{download_root}': {e}")

        print(f"[-] Loading model '{self.model_params['model_size_or_path']}' "
              f"on {self.model_params['device']} ({self.model_params['compute_type']})...")
        
        try:
            self.model = WhisperModel(**self.model_params)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Tip: If using CUDA, make sure cuDNN DLLs are in 'libs' folder.")
            raise e

    def transcribe(self, audio_path: str, language: str = None, **kwargs) -> Generator:
        """
        오디오 파일을 전사합니다.
        :param audio_path: 오디오 파일 경로
        :param language: 언어 코드 (None이면 자동 감지)
        :param kwargs: transcribe 메서드에 전달할 추가 파라미터 (beam_size, initial_prompt 등)
        :return: segments 제너레이터, info 객체
        """
        # 전사 설정 병합
        transcribe_params = DEFAULT_TRANSCRIBE_PARAMS.copy()
        if language:
            transcribe_params["language"] = language
        
        # kwargs로 들어온 옵션 업데이트 (예: initial_prompt)
        transcribe_params.update(kwargs)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        print(f"[-] Transcribing '{audio_path}'...")
        
        # WhisperModel.transcribe 실행
        # segments는 제너레이터이므로 실제 연산은 순회할 때 발생
        return self.model.transcribe(audio_path, **transcribe_params)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """초 단위를 SRT 타임스탬프 형식으로 변환"""
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    @staticmethod
    def save_srt(segments, output_path: str):
        """세그먼트를 SRT 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                start = WhisperEngine.format_timestamp(segment.start)
                end = WhisperEngine.format_timestamp(segment.end)
                text = segment.text.strip()
                
                # 터미널 출력 (선택 사항)
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
                
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        print(f"[-] Saved to {output_path}")


# --- [CLI 실행 부분] ---
def main():
    parser = argparse.ArgumentParser(description="Faster-Whisper CLI & Engine")
    parser.add_argument('input_file', type=str, help='Path to the input audio/video file')
    parser.add_argument('--model', type=str, default='large-v3', help='Model size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--quantization', type=str, default='int8', choices=['int8', 'float16'], help='Compute type')
    parser.add_argument('--lang', type=str, default=None, help='Language code')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for decoding')
    parser.add_argument('--chunk_length', type=int, default=30, help='Chunk length (seconds) for processing')
    
    args = parser.parse_args()

    # 1. 엔진 초기화 (원하는 옵션을 kwargs로 전달 가능)
    try:
        engine = WhisperEngine(
            model_name=args.model,
            device=args.device,
            compute_type=args.quantization
            # cpu_threads=4  <-- 이런 식으로 추가 옵션 전달 가능
        )
    except Exception:
        sys.exit(1)

    # 2. 전사 옵션 설정
    # 한국어일 때만 프롬프트 적용하는 로직 유지
    initial_prompt = None
    if args.lang == 'ko':
        initial_prompt = "이것은 한국어 대화의 자막입니다. 문장 부호를 정확히 입력하세요."

    # 3. 전사 실행
    start_time = time.time()
    
    segments, info = engine.transcribe(
        args.input_file,
        language=args.lang,
        beam_size=args.beam_size,
        chunk_length=args.chunk_length,
        initial_prompt=initial_prompt
        # temperature=0.2 <-- 추가 옵션 전달 가능
    )

    if args.lang is None:
        print(f"[-] Detected language: {info.language} (Probability: {info.language_probability:.2f})")

    # 4. SRT 저장
    base_name = os.path.splitext(args.input_file)[0]
    srt_filename = f"{base_name}.srt"
    
    engine.save_srt(segments, srt_filename)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[-] Done! Processed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
