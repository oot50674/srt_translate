# test.py
import os
import sys
from dotenv import load_dotenv

# .env 로드 설정 (app.py와 유사)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)

from google.genai import types
from module.gemini_module import GeminiClient

class TestGeminiClient(GeminiClient):
    def test_upload_images(
        self,
        image_paths: list[str],
        *,
        prompt: str = "각 이미지에 있는 고양이들의 품종과 특징을 요약해줘",
        model: str | None = None,
    ) -> str:
        """
        여러 이미지 파일을 업로드한 뒤 모델에 한 번에 질의하여 응답 텍스트를 반환합니다.
        내부 히스토리에도 사용자/모델 턴이 추가됩니다.
        """
        parts = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
            print(f"이미지 업로드 중: {image_path}")
            # 파일 업로드
            upload_config = types.UploadFileConfig(displayName=os.path.basename(image_path))
            uploaded = self.client.files.upload(
                file=image_path,
                config=upload_config,
            )
            file_uri = getattr(uploaded, "uri", None) or getattr(uploaded, "file_uri", None)
            mime_type = getattr(uploaded, "mime_type", None) or getattr(uploaded, "mimeType", None)
            if not file_uri or not mime_type:
                raise RuntimeError(f"파일 업로드 결과를 해석하지 못했습니다: {uploaded!r} ({image_path})")

            # 이미지 파트를 parts 리스트에 추가
            parts.append({"file_data": {"file_uri": file_uri, "mime_type": mime_type}})

        # 컨텍스트 압축 및 RPM 체크 (부모 클래스 제공)
        self._prepare_history_for_new_message(prompt or "")
        self._check_rpm_limit()

        # 프롬프트 텍스트를 parts 리스트 마지막에 추가
        if prompt:
            parts.append({"text": prompt})

        user_message = {"role": "user", "parts": parts}
        self.history.append(user_message)

        # 부모의 생성 설정을 그대로 사용
        request_config = self.generation_config.copy()
        config_payload = types.GenerateContentConfig(**request_config)

        # 호출 및 응답 저장
        response = self.client.models.generate_content(
            model=model or self.model,
            contents=self.history,
            config=config_payload,
        )
        text = (response.text or "").strip()
        self.history.append({"role": "model", "parts": [{"text": text}]})
        return text


if __name__ == "__main__":
    # 실행 인자에서 파일 경로들과 마지막 프롬프트를 분리
    if len(sys.argv) < 2:
        print("사용법: python test.py <image_path1> [image_path2 ...] [prompt]")
        sys.exit(1)

    image_paths = []
    prompt = "각 이미지에 있는 고양이들의 품종과 특징을 요약해줘" # 기본 프롬프트

    # 마지막 인자가 파일이 아니면 프롬프트로 간주
    potential_prompt = sys.argv[-1]
    args = sys.argv[1:]
    if not os.path.exists(potential_prompt) and not potential_prompt.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        prompt = potential_prompt
        image_paths = args[:-1]
    else:
        image_paths = args

    if not image_paths:
        print("오류: 하나 이상의 이미지 파일 경로를 제공해야 합니다.")
        sys.exit(1)

    # GOOGLE_API_KEY 환경변수를 사용합니다. (.env에서 로드됨)
    client = TestGeminiClient()
    client.start_chat()
    print(f"총 {len(image_paths)}개 이미지 분석 요청...")
    print(client.test_upload_images(image_paths, prompt=prompt))
