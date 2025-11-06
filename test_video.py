from module.gemini_module import GeminiClient
from google.genai import types
import os
import yt_dlp
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoTestClient(GeminiClient):
    """
    GeminiClient를 상속받아 비디오 인식 테스트 기능을 추가한 클래스
    """

    def prepare_video_parts(self, video_path, max_wait_seconds=300):
        """
        비디오 파일을 업로드하고 file_data 파트를 반환합니다.

        Args:
            video_path (str): 업로드할 비디오 파일 경로
            max_wait_seconds (int): 파일이 ACTIVE 상태가 될 때까지 최대 대기 시간 (초)

        Returns:
            list: file_data 파트 리스트

        Raises:
            FileNotFoundError: 비디오 파일이 존재하지 않을 때
            RuntimeError: 업로드된 파일 메타데이터를 해석하지 못했을 때
            TimeoutError: 파일이 시간 내에 ACTIVE 상태가 되지 않을 때
        """
        import time

        if not os.path.exists(video_path):
            logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

        logger.info(f"비디오 업로드 준비: {video_path}")
        upload_config = types.UploadFileConfig(displayName=os.path.basename(video_path))
        uploaded = self.client.files.upload(file=video_path, config=upload_config)

        file_name = getattr(uploaded, "name", None)
        file_uri = getattr(uploaded, "uri", None) or getattr(uploaded, "file_uri", None)
        mime_type = getattr(uploaded, "mime_type", None) or getattr(uploaded, "mimeType", None)

        if not file_uri or not mime_type:
            logger.error(f"업로드된 파일 메타데이터를 해석하지 못했습니다: {uploaded}")
            raise RuntimeError(f"업로드된 파일 메타데이터를 해석하지 못했습니다: {uploaded!r}")

        logger.info(f"비디오 업로드 완료: {file_uri} (MIME: {mime_type})")

        # 파일이 ACTIVE 상태가 될 때까지 대기
        logger.info(f"파일 처리 대기 중 (최대 {max_wait_seconds}초)...")
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:
                raise TimeoutError(f"파일이 {max_wait_seconds}초 내에 ACTIVE 상태가 되지 않았습니다.")

            # 파일 상태 확인
            file_info = self.client.files.get(name=file_name)
            state = getattr(file_info, "state", None)

            logger.info(f"파일 상태: {state} (경과 시간: {elapsed:.1f}초)")

            if state == "ACTIVE":
                logger.info("파일이 ACTIVE 상태가 되었습니다. 처리를 계속합니다.")
                break
            elif state == "FAILED":
                raise RuntimeError(f"파일 처리가 실패했습니다: {file_info}")

            # 2초 대기 후 재확인
            time.sleep(2)

        return [{"file_data": {"file_uri": file_uri, "mime_type": mime_type}}]

    def test_video_recognition(self, video_path, prompt="이 비디오의 내용을 자세히 설명해주세요."):
        """
        비디오를 업로드하고 내용 인식 테스트를 수행합니다.

        Args:
            video_path (str): 테스트할 비디오 파일 경로
            prompt (str): Gemini에게 질문할 프롬프트

        Returns:
            str: Gemini의 응답
        """
        logger.info(f"비디오 인식 테스트 시작: {video_path}")

        # 비디오 파일 업로드
        parts = self.prepare_video_parts(video_path)

        # 프롬프트 추가
        if prompt:
            parts.append({"text": prompt})

        # Gemini에게 메시지 전송
        response = self.send_message(message_parts=parts)

        logger.info("비디오 인식 테스트 완료")
        return response


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
    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
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
        # VideoTestClient 인스턴스 생성
        client = VideoTestClient(api_key=args.api_key, model="gemini-2.0-flash-exp")
        client.start_chat()

        # 비디오 인식 테스트 실행
        response = client.test_video_recognition(video_path, args.prompt)

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
