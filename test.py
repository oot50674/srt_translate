# test.py
import os
import sys
from dotenv import load_dotenv

# .env 로드 설정 (app.py와 유사)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)

from module.gemini_module import GeminiClient


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("사용법: python test.py <image_path1> [image_path2 ...] [prompt]")
        sys.exit(1)

    args = argv[1:]
    default_prompt = "각 이미지에 있는 고양이들의 품종과 특징을 요약해줘"
    prompt = default_prompt

    potential_prompt = args[-1]
    image_paths = args
    if not os.path.exists(potential_prompt) and not potential_prompt.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        prompt = potential_prompt
        image_paths = args[:-1]

    if not image_paths:
        print("오류: 하나 이상의 이미지 파일 경로를 제공해야 합니다.")
        sys.exit(1)

    client = GeminiClient()
    client.start_chat()
    print(f"총 {len(image_paths)}개 이미지 분석 요청...")
    response = client.send_image_prompt(image_paths, prompt=prompt)
    print(response)


if __name__ == "__main__":
    main(sys.argv)
