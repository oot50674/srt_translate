from module.gemini_module import GeminiClient
import os
import yt_dlp
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_youtube_video(url, output_path='./download_video'):
    """
    yt-dlp를 이용해 유튜브 동영상을 다운로드합니다.

    Args:
        url (str): 유튜브 동영상 URL
        output_path (str): 영상을 저장할 디렉터리

    Returns:
        str: 다운로드된 비디오 파일 경로 (실패 시 None)
    """
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'format': 'best[height<=720]',  # 720p 이하 중 가장 좋은 화질로 다운로드
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {url}")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"Video downloaded successfully: {filename}")
            return filename
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='유튜브 비디오를 다운로드하고 Gemini로 인식 테스트')
    parser.add_argument('--url', type=str, default='https://youtu.be/L0WyWY4pmvs?si=nc9Q7q3ap9RSDnDN',
                       help='유튜브 동영상 URL (기본값: Rick Astley - Never Gonna Give You Up)')
    parser.add_argument('-o', '--output', type=str, default='./download_video',
                       help='출력 디렉터리 (기본값: ./download_video)')
    parser.add_argument('--prompt', type=str, default='이 비디오의 내용을 자세히 설명해주세요.',
                       help='Gemini에게 질문할 프롬프트')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Gemini API 키 (없으면 환경변수 GOOGLE_API_KEY 사용)')
    parser.add_argument('--video-path', type=str, default=None,
                       help='이미 다운로드된 비디오 파일 경로 (지정하면 다운로드 건너뜀)')

    args = parser.parse_args()

    # 1. 유튜브 비디오 다운로드 (또는 기존 파일 사용)
    if args.video_path and os.path.exists(args.video_path):
        print(f"Using existing video: {args.video_path}")
        video_path = args.video_path
    else:
        video_path = download_youtube_video(args.url, args.output)

        if not video_path or not os.path.exists(video_path):
            print("비디오 다운로드 실패")
            exit(1)

    # 2. Gemini로 비디오 인식 테스트
    print(f"\n{'='*60}")
    print("Testing video recognition with Gemini...")
    print(f"{'='*60}\n")

    try:
        # GeminiClient 인스턴스 생성
        client = GeminiClient(api_key=args.api_key, model="gemini-2.0-flash-exp")
        client.start_chat()

        # 비디오 인식 테스트 실행 (통합 메서드 send_message 사용)
        response = client.send_message(args.prompt, media_paths=video_path)

        # 결과 출력
        print(f"\n{'='*60}")
        print("=== Gemini 응답 ===")
        print(f"{'='*60}\n")
        print(response)
        print(f"\n{'='*60}")

    except Exception as e:
        print(f"\nError testing video recognition: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
