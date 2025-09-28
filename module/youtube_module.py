#!/usr/bin/env python3
"""
YouTube 자막 추출 모듈
최신 youtube-transcript-api (v1.2.2)를 사용하여 Flask 웹 애플리케이션에 통합
"""

import re
import os
import time
import logging
from typing import Optional, List, Dict, Any
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
    RequestBlocked,
    IpBlocked,
    YouTubeRequestFailed,
)

try:
    from youtube_transcript_api.proxies import WebshareProxyConfig
    PROXY_SUPPORT = True
except ImportError:
    PROXY_SUPPORT = False

# 상수 정의
TARGET_LANGUAGE = 'ko'
SOURCE_LANGUAGES = ['en', 'ko']
VIDEO_ID_LENGTH = 11
RATE_LIMIT_DELAY = 3.0  # 요청 간 대기 시간 (초)

logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]:
    """
    YouTube URL에서 비디오 ID를 추출합니다.
    
    지원하는 URL 형식:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://www.youtube.com/watch?v=VIDEO_ID&t=30s
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - VIDEO_ID (11자리 비디오 ID)
    
    Args:
        url: YouTube URL 또는 비디오 ID
        
    Returns:
        추출된 11자리 비디오 ID, 실패시 None
    """
    if not url or not isinstance(url, str):
        logger.error("유효하지 않은 URL입니다.")
        return None
        
    url = url.strip()
    
    # 직접 비디오 ID가 입력된 경우
    if (len(url) == VIDEO_ID_LENGTH and 
        not (url.startswith('http') or url.startswith('www')) and
        re.match(r'^[a-zA-Z0-9_-]+$', url)):
        logger.info(f"비디오 ID 직접 입력 감지: {url}")
        return url
    
    # YouTube URL 패턴들
    patterns = [
        r"[?&]v=([a-zA-Z0-9_-]{11})",  # watch?v= 형식
        r"embed/([a-zA-Z0-9_-]{11})",   # embed/ 형식
        r"youtu\.be/([a-zA-Z0-9_-]{11})" # youtu.be 형식
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.info(f"URL에서 비디오 ID 추출 성공: {video_id}")
            return video_id
    
    logger.error(f"비디오 ID를 추출할 수 없습니다: {url}")
    return None

def get_transcript(video_id: str, target_lang: str = TARGET_LANGUAGE, languages: List[str] = None) -> Dict[str, Any]:
    """
    YouTube 비디오의 자막을 가져오고 필요시 번역합니다.
    최신 youtube-transcript-api (v1.2.2) 사용
    
    Args:
        video_id: YouTube 비디오 ID
        target_lang: 목표 언어 코드 (기본값: 'ko')
        languages: 우선 언어 목록 (기본값: ['ko', 'en', 'ja', 'zh', 'es', 'fr', 'de'])
        
    Returns:
        Dict containing transcript data and metadata
        
    Raises:
        VideoUnavailable: 비디오를 찾을 수 없음
        TranscriptsDisabled: 자막이 비활성화됨
        NoTranscriptFound: 사용 가능한 자막이 없음
        CouldNotRetrieveTranscript: 자막 검색 실패
    """
    # 레이트 리미팅 적용
    time.sleep(RATE_LIMIT_DELAY)
    
    # 기본 언어 우선순위 설정
    if languages is None:
        languages = [target_lang, 'en', 'ja', 'zh', 'es', 'fr', 'de', 'it', 'pt', 'ru']
    
    # 프록시 설정 생성 (환경변수에서)
    proxy_config = create_proxy_config()
    
    try:
        # YouTube Transcript API 인스턴스 생성
        if proxy_config and PROXY_SUPPORT:
            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
            logger.info("프록시 설정을 사용하여 자막을 가져옵니다.")
        else:
            ytt_api = YouTubeTranscriptApi()
            
        # 자막 목록 가져오기
        transcript_list = ytt_api.list(video_id)
        logger.info(f"비디오 ID {video_id}의 자막 목록을 가져왔습니다.")

        # 목표 언어의 자막 직접 검색 (수동 우선)
        try:
            transcript = transcript_list.find_transcript([target_lang])
            fetched_transcript = transcript.fetch()
            logger.info(f"{target_lang} 자막을 직접 발견했습니다 (생성 유형: {'수동' if not transcript.is_generated else '자동'})")
            return {
                'transcript_data': fetched_transcript.to_raw_data(),
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'source_transcript': f"{transcript.language} ({'자동' if transcript.is_generated else '수동'})"
            }
        except NoTranscriptFound:
            logger.debug(f"목표 언어 {target_lang} 자막이 없습니다.")

        # 다양한 언어의 자막 검색 및 번역 시도
        for lang_code in languages:
            if lang_code == target_lang:
                continue  # 이미 시도했음
                
            try:
                transcript = transcript_list.find_transcript([lang_code])
                
                # 번역이 가능한지 확인
                if transcript.is_translatable and target_lang in [t.language_code for t in transcript.translation_languages]:
                    try:
                        logger.info(f"{transcript.language}({lang_code}) 자막을 {target_lang}으로 번역 중...")
                        translated_transcript = transcript.translate(target_lang)
                        fetched_transcript = translated_transcript.fetch()
                        
                        return {
                            'transcript_data': fetched_transcript.to_raw_data(),
                            'language': translated_transcript.language,
                            'language_code': translated_transcript.language_code,
                            'is_generated': transcript.is_generated,
                            'source_transcript': f"{transcript.language} ({'자동' if transcript.is_generated else '수동'}) → {target_lang} 번역"
                        }
                    except Exception as translate_err:
                        logger.warning(f"{lang_code} → {target_lang} 번역 실패: {translate_err}")
                        continue
                else:
                    # 번역이 불가능하면 원본 언어 그대로 사용
                    if lang_code in ['en']:  # 영어는 그대로 사용 가능
                        fetched_transcript = transcript.fetch()
                        logger.info(f"{transcript.language}({lang_code}) 자막을 원본 그대로 사용합니다.")
                        
                        return {
                            'transcript_data': fetched_transcript.to_raw_data(),
                            'language': transcript.language,
                            'language_code': transcript.language_code,
                            'is_generated': transcript.is_generated,
                            'source_transcript': f"{transcript.language} ({'자동' if transcript.is_generated else '수동'})"
                        }
                        
            except NoTranscriptFound:
                logger.debug(f"{lang_code} 자막이 없습니다.")
                continue

        # 사용 가능한 첫 번째 자막 사용 (번역 없이)
        available_transcripts = list(transcript_list)
        if available_transcripts:
            first_transcript = available_transcripts[0]
            fetched_transcript = first_transcript.fetch()
            
            logger.info(f"첫 번째 사용 가능한 자막을 사용합니다: {first_transcript.language}")
            return {
                'transcript_data': fetched_transcript.to_raw_data(),
                'language': first_transcript.language,
                'language_code': first_transcript.language_code,
                'is_generated': first_transcript.is_generated,
                'source_transcript': f"{first_transcript.language} ({'자동' if first_transcript.is_generated else '수동'})"
            }

        # 사용 가능한 자막이 없음
        raise NoTranscriptFound(video_id, [], [])

    except YouTubeRequestFailed as e:
        logger.error(f"YouTube 요청 실패: {e}")
        raise
    except RequestBlocked as e:
        logger.error(f"요청이 차단됨: {e}")
        raise
    except IpBlocked as e:
        logger.error(f"IP가 차단됨: {e}")
        raise
    except VideoUnavailable as e:
        logger.error(f"비디오 접근 불가: {e}")
        raise
    except TranscriptsDisabled as e:
        logger.error(f"자막이 비활성화됨: {e}")
        raise
    except CouldNotRetrieveTranscript as e:
        logger.error(f"자막 검색 실패: {e}")
        raise
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise

def create_proxy_config():
    """
    환경변수에서 프록시 설정을 읽어 프록시 구성을 생성합니다.
    
    Returns:
        프록시 설정 객체 또는 None
    """
    if not PROXY_SUPPORT:
        return None
        
    # 환경변수에서 WebShare 프록시 설정 읽기
    proxy_username = os.getenv('WEBSHARE_USERNAME')
    proxy_password = os.getenv('WEBSHARE_PASSWORD') 
    
    if all([proxy_username, proxy_password]):
        try:
            return WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password
            )
        except Exception as e:
            logger.warning(f"프록시 설정 실패: {e}")
            return None
    
    return None

def format_transcript_text(text: str) -> str:
    """
    자막 텍스트의 가독성을 향상시킵니다.
    
    Args:
        text: 원본 자막 텍스트
        
    Returns:
        정리된 자막 텍스트
    """
    # 불필요한 공백 제거
    text = ' '.join(text.split())
    
    # 음악 표기나 배경음 표기 제거
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 특수문자 정리
    text = text.replace('\n', ' ').strip()
    
    return text if text else '[음성 없음]'

def format_time_srt(seconds: float) -> str:
    """
    초 단위 시간을 SRT 형식의 타임스탬프로 변환합니다.
    
    Args:
        seconds: 초 단위 시간 (float)
        
    Returns:
        SRT 형식의 타임스탬프 (HH:MM:SS,mmm)
    """
    try:
        total_seconds = float(seconds)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    except (ValueError, TypeError):
        return "00:00:00,000"

def transcript_to_srt(transcript_data: List[Dict[str, Any]]) -> str:
    """
    자막 데이터를 SRT 형식으로 변환합니다.
    
    Args:
        transcript_data: 자막 데이터 리스트 (youtube-transcript-api v1.2.2 형식)
        
    Returns:
        SRT 형식의 자막 문자열
    """
    srt_content = []
    valid_entries = 0
    
    for entry in transcript_data:
        text = format_transcript_text(entry.get("text", ""))
        
        # 빈 텍스트 건너뛰기
        if text == '[음성 없음]' or not text.strip():
            continue
            
        valid_entries += 1
        start_time = entry.get("start", 0.0)
        duration = entry.get("duration", 0.0)
        end_time = start_time + duration
        
        start_srt = format_time_srt(start_time)
        end_srt = format_time_srt(end_time)
        
        srt_entry = f"{valid_entries}\n{start_srt} --> {end_srt}\n{text}\n"
        srt_content.append(srt_entry)
    
    return "\n".join(srt_content)

def extract_youtube_transcript(url: str, target_lang: str = 'ko', languages: List[str] = None) -> Dict[str, Any]:
    """
    YouTube URL에서 자막을 추출하여 SRT 형식으로 변환합니다.
    최신 youtube-transcript-api (v1.2.2) 사용
    
    Args:
        url: YouTube URL 또는 비디오 ID
        target_lang: 목표 언어 코드 (기본값: 'ko')
        languages: 우선 언어 목록 (기본값: None)
        
    Returns:
        성공시: {
            "success": True, 
            "video_id": str, 
            "srt_content": str, 
            "transcript_count": int,
            "source_transcript": str,
            "language": str,
            "language_code": str
        }
        실패시: {"success": False, "error": str, "error_type": str}
    """
    try:
        # 비디오 ID 추출
        video_id = extract_video_id(url)
        if not video_id:
            return {
                "success": False,
                "error": "유효하지 않은 YouTube URL 또는 비디오 ID입니다.",
                "error_type": "invalid_url"
            }
        
        logger.info(f"자막 추출 시작: 비디오 ID {video_id}, 목표 언어: {target_lang}")
        
        # 자막 가져오기 (개선된 다국어 지원)
        transcript_result = get_transcript(video_id, target_lang, languages)
        
        # SRT 형식으로 변환
        srt_content = transcript_to_srt(transcript_result['transcript_data'])
        
        transcript_count = len(transcript_result['transcript_data'])
        logger.info(f"자막 추출 완료: {transcript_count}개 항목, 소스: {transcript_result['source_transcript']}")
        
        return {
            "success": True,
            "video_id": video_id,
            "srt_content": srt_content,
            "transcript_count": transcript_count,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "source_transcript": transcript_result['source_transcript'],
            "language": transcript_result['language'],
            "language_code": transcript_result['language_code'],
            "is_generated": transcript_result['is_generated']
        }
        
    except YouTubeRequestFailed:
        return {
            "success": False,
            "error": "YouTube 요청이 실패했습니다. 잠시 후 다시 시도하세요.",
            "error_type": "request_failed"
        }
    except RequestBlocked:
        return {
            "success": False,
            "error": "요청이 차단되었습니다. 잠시 후 다시 시도하세요.",
            "error_type": "request_blocked"
        }
    except IpBlocked:
        return {
            "success": False,
            "error": "IP 주소가 차단되었습니다. 잠시 후 다시 시도하세요.",
            "error_type": "ip_blocked"
        }
    except VideoUnavailable:
        return {
            "success": False,
            "error": "비디오를 찾을 수 없거나 접근할 수 없습니다. URL을 확인하세요.",
            "error_type": "video_unavailable"
        }
    except TranscriptsDisabled:
        return {
            "success": False,
            "error": "이 비디오는 자막이 비활성화되어 있습니다.",
            "error_type": "transcripts_disabled"
        }
    except NoTranscriptFound:
        return {
            "success": False,
            "error": "사용 가능한 자막이 없습니다.",
            "error_type": "no_transcript"
        }
    except CouldNotRetrieveTranscript as e:
        return {
            "success": False,
            "error": f"자막을 가져올 수 없습니다: {str(e)}",
            "error_type": "retrieve_failed"
        }
    except Exception as e:
        logger.exception("예상치 못한 오류:")
        return {
            "success": False,
            "error": f"처리 중 오류가 발생했습니다: {str(e)}",
            "error_type": "unexpected_error"
        }