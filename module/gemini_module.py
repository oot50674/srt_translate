import os
import time
from collections import deque
from typing import Iterator, Dict, Any, Optional, List, Union, cast
import logging
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Google Gen AI SDK를 사용하는 대화형 Gemini API 클라이언트.
    대화 히스토리를 관리하고, 문매을 유지하며, 대화 제어 기능을 제공합니다.
    RPM(분당 요청 수) 제한 기능과 JSON 스키마 응답 기능도 포함합니다.
    """
    def __init__(self, 
                 model: str = "gemini-2.5-flash-lite-preview-06-17",
                 api_key: Optional[str] = None,
                 generation_config: Optional[Dict[str, Any]] = None,
                 rpm_limit: int = 10,
                 safety_settings: Optional[List[Dict[str, str]]] = None,
                 thinking_budget: int = 8192,
                 response_schema: Optional[Dict] = None,
                 response_mime_type: Optional[str] = None,
                 context_compression_enabled: bool = False,
                 context_limit_tokens: Optional[int] = None,
                 context_keep_recent: int = 2,
                 **kwargs):
        """
        대화형 GeminiClient를 초기화합니다.

        Args:
            model (str): 사용할 기본 모델 이름.
            api_key (str, optional): API 키. 없으면 환경변수에서 로드합니다.
            generation_config (dict, optional): 생성 설정 (temperature, max_output_tokens 등).
            rpm_limit (int): 분당 최대 요청 수 (기본값: 10).
            safety_settings (list, optional): 안전 설정 목록.
            thinking_budget (int): 사고 과정 토큰 예산 (기본값: 8192).
            response_schema (dict, optional): JSON 응답 스키마.
            response_mime_type (str, optional): 응답 MIME 타입 (기본값: response_schema 사용시 "application/json").
            context_compression_enabled (bool): 컨텍스트 압축 사용 여부.
            context_limit_tokens (int, optional): 히스토리 토큰 제한 (추정치 기반).
            context_keep_recent (int): 압축 시 유지할 최신 턴 수.
            **kwargs: 추가 설정 옵션들.
        
        Raises:
            ValueError: API 키를 찾을 수 없는 경우.
        """
        # API 키 설정 - 전달된 값을 우선 사용하고, 없으면 환경변수로 대체
        if api_key:
            api_key = api_key.strip()
        
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("API 키가 제공되지 않았습니다. 웹 인터페이스에서 API 키를 입력하거나 GOOGLE_API_KEY 환경변수를 설정하세요.")
                raise ValueError("Gemini API 키가 필요합니다. 웹 인터페이스에서 API 키를 입력하거나 GOOGLE_API_KEY를 설정하세요.")
            else:
                logger.info("환경변수(GOOGLE_API_KEY)에서 API 키를 로드했습니다.")
        else:
            logger.info("사용자가 제공한 API 키를 사용합니다.")

        self.api_key = api_key
        
        # response_schema 유효성 검사
        if response_schema is not None:
            if isinstance(response_schema, dict):
                logger.info("딕셔너리 스키마가 설정되었습니다.")
            else:
                logger.warning("지원되지 않는 스키마 타입입니다. 딕셔너리 스키마를 사용하세요.")
        
        # response_mime_type 자동 설정
        if response_schema is not None and response_mime_type is None:
            response_mime_type = "application/json"
            logger.info("response_schema 사용으로 response_mime_type이 'application/json'으로 자동 설정되었습니다.")
        
        # 기본 safety_settings 생성 (모든 카테고리 BLOCK_NONE)
        if safety_settings is None:
            default_categories = [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in default_categories]
        
        try:
            self.client = genai.Client(api_key=api_key)
            self.model = model
            
            # 대화 히스토리 초기화
            self.history: List[Dict[str, Any]] = []

            # 컨텍스트 압축 설정
            self.context_compression_enabled = bool(context_compression_enabled)
            if context_limit_tokens is not None and context_limit_tokens <= 0:
                context_limit_tokens = None
            self.context_limit_tokens = context_limit_tokens
            self.context_keep_recent = max(0, int(context_keep_recent))
            self._context_summary_prefix = "[컨텍스트 요약]"
            
            # RPM 제한 관련 초기화
            self.rpm_limit = rpm_limit
            self.request_times = deque()  # 최근 요청 시간들을 저장
            
            # JSON 스키마 관련 설정 저장
            self.default_response_schema = response_schema
            self.default_response_mime_type = response_mime_type
            
            # 생성 설정 초기화
            default_generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 1048576,
                'top_p': 0.9,
                'top_k': 40,
                'safety_settings': safety_settings,
                'thinking_config': types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                ),
            }
            
            # JSON 스키마 설정이 있으면 추가
            if response_schema is not None:
                default_generation_config['response_schema'] = response_schema
            if response_mime_type is not None:
                default_generation_config['response_mime_type'] = response_mime_type
            
            if generation_config:
                default_generation_config.update(generation_config)
            default_generation_config.update(kwargs)
            self.generation_config = default_generation_config
            
            # fork를 위한 초기화 인자 저장
            self._init_args = {
                'model': model,
                'api_key': api_key,
                'generation_config': self.generation_config.copy(),
                'rpm_limit': rpm_limit,
                'safety_settings': safety_settings,
                'thinking_budget': thinking_budget,
                'response_schema': response_schema,
                'response_mime_type': response_mime_type,
                'context_compression_enabled': self.context_compression_enabled,
                'context_limit_tokens': self.context_limit_tokens,
                'context_keep_recent': self.context_keep_recent,
            }
            
            logger.info(f"대화형 Gemini 클라이언트가 초기화되었습니다. 모델: {self.model}")
            logger.info(f"RPM 제한: {self.rpm_limit}")
            logger.info(f"Thinking Budget: {thinking_budget}")
            if response_schema:
                logger.info(f"JSON 스키마 모드가 활성화되었습니다.")
            logger.info(f"생성 설정: {self.generation_config}")
        except Exception as e:
            logger.error(f"Gemini 클라이언트 초기화 중 오류 발생: {e}")
            raise

    def _check_rpm_limit(self):
        """
        RPM 제한을 확인하고 필요시 대기합니다.
        """
        current_time = time.time()
        
        # 1분(60초) 이전의 요청들을 제거
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # 현재 분당 요청 수가 제한을 초과하는지 확인
        if len(self.request_times) >= self.rpm_limit:
            # 가장 오래된 요청으로부터 60초 후까지 대기
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"RPM 제한 ({self.rpm_limit})에 도달했습니다. {wait_time:.2f}초 대기합니다.")
                time.sleep(wait_time)
                # 대기 후 다시 정리
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > 60:
                    self.request_times.popleft()
        
        # 현재 요청 시간 기록
        self.request_times.append(current_time)

    def _estimate_text_tokens(self, text: str) -> int:
        """간단한 휴리스틱으로 텍스트 토큰 수를 추정합니다."""
        if not text:
            return 0
        text = text.strip()
        if not text:
            return 0
        char_tokens = max(1, len(text) // 4)
        word_tokens = len(text.split())
        return max(char_tokens, word_tokens, 1)

    def _estimate_history_tokens(self) -> int:
        """현재 히스토리의 토큰 수를 추정합니다."""
        total = 0
        for message in self.history:
            parts = message.get('parts') or []
            for part in parts:
                total += self._estimate_text_tokens(part.get('text', ''))
        total += len(self.history) * 4  # 역할/메타데이터 오버헤드 보정
        return total

    def _prepare_history_for_new_message(self, upcoming_text: str) -> None:
        """새 메시지를 추가하기 전에 컨텍스트 압축이 필요하면 수행합니다."""
        if not self.context_compression_enabled or not self.context_limit_tokens:
            return
        try:
            self._compress_history_if_needed(upcoming_text or '')
        except Exception as exc:
            logger.warning("컨텍스트 압축 중 오류가 발생해 기존 히스토리를 유지합니다: %s", exc)

    def _compress_history_if_needed(self, upcoming_text: str) -> None:
        """히스토리가 제한을 초과하면 요약을 통해 압축합니다."""
        limit = self.context_limit_tokens
        if not limit:
            return

        upcoming_tokens = self._estimate_text_tokens(upcoming_text)
        attempts = 0
        keep_turns = self.context_keep_recent

        while attempts < 3:
            current_tokens = self._estimate_history_tokens()
            projected_tokens = current_tokens + upcoming_tokens
            if projected_tokens <= limit:
                logger.debug(
                    "컨텍스트 압축 불필요: 히스토리 %sT + 다음 %sT <= 제한 %sT",
                    current_tokens,
                    upcoming_tokens,
                    limit,
                )
                return

            prev_keep_turns = keep_turns
            logger.info(
                "컨텍스트 압축 시도 %s회차: 히스토리 %sT, 다음 %sT, 제한 %sT, keep_recent=%s",
                attempts + 1,
                current_tokens,
                upcoming_tokens,
                limit,
                prev_keep_turns,
            )

            compressed = self._compress_history_once(prev_keep_turns)
            if not compressed:
                logger.warning(
                    "컨텍스트 압축 중단: 히스토리 %sT, 다음 %sT, 제한 %sT (keep_recent=%s)",
                    current_tokens,
                    upcoming_tokens,
                    limit,
                    prev_keep_turns,
                )
                return

            attempts += 1
            new_tokens = self._estimate_history_tokens()
            logger.info(
                "컨텍스트 압축 결과 %s회차: 히스토리 %sT → %sT (추정)",
                attempts,
                current_tokens,
                new_tokens,
            )

            if new_tokens + upcoming_tokens <= limit:
                return

            keep_turns = max(0, prev_keep_turns - 1)

        final_tokens = self._estimate_history_tokens()
        final_projected = final_tokens + upcoming_tokens
        if final_projected > limit:
            logger.warning(
                "컨텍스트 압축 한계: 히스토리 %sT → %sT, 다음 %sT 포함 시 %sT / 제한 %sT",
                current_tokens,
                final_tokens,
                upcoming_tokens,
                final_projected,
                limit,
            )

    def _compress_history_once(self, keep_turns: int) -> bool:
        """히스토리의 오래된 부분을 요약 메시지로 대체합니다."""
        if not self.history:
            return False

        keep_turns = max(0, int(keep_turns))
        keep_messages = keep_turns * 2  # 사용자/모델 한 턴당 2개 메시지
        if keep_messages >= len(self.history):
            return False

        if keep_messages % 2 == 1:
            keep_messages = min(len(self.history), keep_messages + 1)

        split_index = len(self.history) - keep_messages if keep_messages else len(self.history)
        compress_candidates = self.history[:split_index]
        preserved_history = self.history[split_index:] if split_index < len(self.history) else []

        if not compress_candidates:
            return False

        summary_text = self._summarize_messages(compress_candidates)
        if not summary_text:
            return False

        summary_message = {
            'role': 'model',
            'parts': [{'text': f"{self._context_summary_prefix}\n{summary_text}"}]
        }

        new_history: List[Dict[str, Any]] = [summary_message]
        new_history.extend(preserved_history)
        self.history = new_history
        logger.info("대화 히스토리가 요약되어 압축되었습니다. 현재 메시지 수: %s", len(self.history))
        return True

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """지정된 메시지 구간을 요약합니다."""
        history_dump_lines: List[str] = []
        for msg in messages:
            role = msg.get('role', 'user').upper()
            parts = msg.get('parts') or []
            text_chunks = []
            for part in parts:
                text = (part.get('text') or '').strip()
                if text:
                    text_chunks.append(text)
            if not text_chunks:
                continue
            joined = '\n'.join(text_chunks).strip()
            if joined:
                history_dump_lines.append(f"{role}:\n{joined}")

        if not history_dump_lines:
            return None

        history_dump = "\n\n---\n\n".join(history_dump_lines)
        summary_prompt = (
            "당신은 번역 세션의 컨텍스트를 요약하는 보조 도구입니다. "
            "아래 대화에서 스타일 규칙, 용어, 주요 결정, 주의/금지 사항, 미해결 항목을 자세하게 정리하세요.\n"
            "응답 형식:\n"
            "- 스타일: ...\n"
            "- 용어: ...\n"
            "- 결정 사항: ...\n"
            "- 주의/금지: ...\n"
            "각 항목이 비어 있으면 '없음'이라고 표기하고, 반드시 한국어로 답변하세요.\n"
            "이전 [컨텍스트 요약]이 히스토리에 존재할 경우 이를 반영하여 작성하세요.\n"
            "=== 대화 기록 시작 ===\n"
            f"{history_dump}\n"
            "=== 대화 기록 끝 ==="
        )

        try:
            self._check_rpm_limit()
            compression_config = {
                k: v for k, v in self.generation_config.items()
                if k not in {'response_schema', 'response_mime_type'}
            }
            compression_config['temperature'] = 0.3
            try:
                max_tokens = int(self.generation_config.get('max_output_tokens', 4096))
            except (TypeError, ValueError):
                max_tokens = 4096
            compression_config['max_output_tokens'] = min(2048, max_tokens)
            config_payload = cast(types.GenerateContentConfigDict, compression_config)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{'role': 'user', 'parts': [{'text': summary_prompt}]}],
                config=config_payload
            )
            summary_text = (response.text or '').strip()
            return summary_text or None
        except Exception as exc:
            logger.warning("컨텍스트 요약 생성 실패: %s", exc)
            return None

    def set_rpm_limit(self, rpm_limit: int):
        """
        RPM 제한을 설정합니다.
        
        Args:
            rpm_limit (int): 새로운 분당 최대 요청 수.
        
        Raises:
            ValueError: rpm_limit가 양수가 아닌 경우.
        """
        if rpm_limit <= 0:
            raise ValueError("RPM 제한은 양수여야 합니다.")
        
        self.rpm_limit = rpm_limit
        self._init_args['rpm_limit'] = rpm_limit
        logger.info(f"RPM 제한이 {rpm_limit}으로 설정되었습니다.")

    def get_current_rpm_usage(self) -> Dict[str, Any]:
        """
        현재 RPM 사용량 정보를 반환합니다.
        
        Returns:
            dict: RPM 사용량 정보 (limit, current_count, time_until_reset).
        """
        current_time = time.time()
        
        # 1분 이전의 요청들을 제거
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        current_count = len(self.request_times)
        time_until_reset = 60 - (current_time - self.request_times[0]) if self.request_times else 0
        
        return {
            'limit': self.rpm_limit,
            'current_count': current_count,
            'remaining': max(0, self.rpm_limit - current_count),
            'time_until_reset': max(0, time_until_reset)
        }

    def start_chat(self, history: Optional[List[Dict[str, Any]]] = None):
        """
        새 대화를 시작하거나 기존 히스토리로 초기화합니다.
        
        Args:
            history (list, optional): 기존 대화 히스토리. None이면 빈 히스토리로 시작.
        """
        self.history = history if history is not None else []
        logger.info(f"새 대화가 시작되었습니다. 히스토리 길이: {len(self.history)}")

    def send_message(self, message: str, model: Optional[str] = None, 
                    response_schema: Optional[Dict] = None,
                    response_mime_type: Optional[str] = None) -> str:
        """
        메시지를 보내고 모델의 응답을 받습니다. 대화 히스토리를 자동으로 관리합니다.

        Args:
            message (str): 사용자 메시지.
            model (str, optional): 사용할 모델. 지정하지 않으면 기본 모델 사용.
            response_schema (dict, optional): 이번 요청만을 위한 JSON 응답 스키마.
            response_mime_type (str, optional): 이번 요청만을 위한 응답 MIME 타입.

        Returns:
            str: 모델의 응답 텍스트.
        
        Raises:
            google_exceptions.GoogleAPICallError: API 호출에 실패한 경우.
            Exception: 그 외 예상치 못한 오류가 발생한 경우.
        """
        # 컨텍스트 압축 사전 처리 및 RPM 확인
        self._prepare_history_for_new_message(message)
        self._check_rpm_limit()
        
        # 사용자 메시지를 히스토리에 추가
        user_message = {
            'role': 'user',
            'parts': [{'text': message}]
        }
        self.history.append(user_message)
        
        try:
            # 이번 요청을 위한 설정 준비
            request_config = self.generation_config.copy()
            
            # 임시 스키마 설정이 있으면 적용
            if response_schema is not None:
                request_config['response_schema'] = response_schema
                if response_mime_type is None:
                    response_mime_type = "application/json"
            
            if response_mime_type is not None:
                request_config['response_mime_type'] = response_mime_type
            
            # 전체 히스토리를 contents로 전달
            config_payload = cast(types.GenerateContentConfigDict, request_config)
            response = self.client.models.generate_content(
                model=model or self.model,
                contents=self.history,
                config=config_payload
            )
            
            # 모델 응답을 히스토리에 추가
            model_response = {
                'role': 'model',
                'parts': [{'text': response.text}]
            }
            self.history.append(model_response)
            
            logger.info(f"메시지 전송 완료. 현재 히스토리 길이: {len(self.history)}")
            return response.text or ""
            
        except google_exceptions.GoogleAPICallError as e:
            # 오류 발생 시 추가한 사용자 메시지 롤백
            self.history.pop()
            logger.error(f"Google API 호출 오류: {e}")
            raise
        except Exception as e:
            # 오류 발생 시 추가한 사용자 메시지 롤백
            self.history.pop()
            logger.error(f"메시지 전송 중 오류: {e}")
            raise

    def send_message_stream(self, message: str, model: Optional[str] = None,
                          response_schema: Optional[Dict] = None,
                          response_mime_type: Optional[str] = None) -> Iterator[str]:
        """
        메시지를 보내고 모델의 응답을 스트리밍으로 받습니다.

        Args:
            message (str): 사용자 메시지.
            model (str, optional): 사용할 모델. 지정하지 않으면 기본 모델 사용.
            response_schema (dict, optional): 이번 요청만을 위한 JSON 응답 스키마.
            response_mime_type (str, optional): 이번 요청만을 위한 응답 MIME 타입.

        Yields:
            str: 생성된 텍스트 조각.
        
        Raises:
            google_exceptions.GoogleAPICallError: API 호출에 실패한 경우.
            Exception: 그 외 예상치 못한 오류가 발생한 경우.
        """
        # 컨텍스트 압축 사전 처리 및 RPM 확인
        self._prepare_history_for_new_message(message)
        self._check_rpm_limit()
        
        # 사용자 메시지를 히스토리에 추가
        user_message = {
            'role': 'user',
            'parts': [{'text': message}]
        }
        self.history.append(user_message)
        
        try:
            # 이번 요청을 위한 설정 준비
            request_config = self.generation_config.copy()
            
            # 임시 스키마 설정이 있으면 적용
            if response_schema is not None:
                request_config['response_schema'] = response_schema
                if response_mime_type is None:
                    response_mime_type = "application/json"
            
            if response_mime_type is not None:
                request_config['response_mime_type'] = response_mime_type
            
            # 전체 히스토리를 contents로 전달하여 스트리밍
            config_payload = cast(types.GenerateContentConfigDict, request_config)
            stream = self.client.models.generate_content_stream(
                model=model or self.model,
                contents=self.history,
                config=config_payload
            )
            
            # 스트리밍 응답 수집
            full_response_text = []
            for chunk in stream:
                if chunk.text:
                    full_response_text.append(chunk.text)
                    yield chunk.text
            
            # 전체 응답을 히스토리에 추가
            model_response = {
                'role': 'model',
                'parts': [{'text': ''.join(full_response_text)}]
            }
            self.history.append(model_response)
            
            logger.info(f"스트리밍 메시지 전송 완료. 현재 히스토리 길이: {len(self.history)}")
            
        except google_exceptions.GoogleAPICallError as e:
            # 오류 발생 시 추가한 사용자 메시지 롤백
            self.history.pop()
            logger.error(f"Google API 호출 오류(스트리밍): {e}")
            raise
        except Exception as e:
            # 오류 발생 시 추가한 사용자 메시지 롤백
            self.history.pop()
            logger.error(f"스트리밍 중 오류: {e}")
            raise

    def delete_last_turn(self) -> bool:
        """
        마지막 대화 턴(사용자 메시지 + 모델 응답)을 삭제합니다.
        
        Returns:
            bool: 삭제 성공 여부.
        """
        if len(self.history) < 2:
            logger.warning("삭제할 대화 턴이 충분하지 않습니다.")
            return False
        
        # 마지막 모델 응답과 사용자 메시지 삭제
        self.history.pop()  # model response
        self.history.pop()  # user message
        logger.info("마지막 대화 턴이 삭제되었습니다.")
        return True

    def fork(self) -> 'GeminiClient':
        """
        현재 대화 히스토리를 복사한 새로운 클라이언트 인스턴스를 생성합니다.
        RPM 제한 설정도 복사됩니다.
        
        Returns:
            GeminiClient: 복제된 클라이언트 인스턴스.
        """
        logger.info("대화를 새 클라이언트 인스턴스로 포크합니다.")
        new_client = GeminiClient(**self._init_args)
        new_client.start_chat(history=[msg.copy() for msg in self.history])  # Deep copy
        return new_client

    def modify_message(self, index: int, new_message: str) -> Optional['GeminiClient']:
        """
        특정 인덱스의 사용자 메시지를 수정하고, 그 지점부터 포크된 새 클라이언트를 반환합니다.
        
        Args:
            index (int): 수정할 메시지의 인덱스.
            new_message (str): 새로운 메시지 내용.
            
        Returns:
            GeminiClient: 수정된 히스토리를 가진 새 클라이언트. 실패 시 None.
        """
        if not (0 <= index < len(self.history)):
            logger.error(f"잘못된 인덱스: {index}")
            return None
        
        if self.history[index].get('role') != 'user':
            logger.error("사용자 메시지만 수정할 수 있습니다.")
            return None
        
        # 수정된 히스토리 생성 (index까지만 포함)
        modified_history = []
        for i in range(index + 1):
            msg = self.history[i].copy()
            if i == index:
                msg['parts'] = [{'text': new_message}]
            modified_history.append(msg)
        
        # 새 클라이언트 생성
        new_client = GeminiClient(**self._init_args)
        new_client.start_chat(history=modified_history)
        logger.info(f"인덱스 {index}의 메시지를 수정하여 대화를 포크했습니다.")
        return new_client

    def get_history(self) -> List[Dict[str, Any]]:
        """현재 대화 히스토리를 반환합니다."""
        return self.history.copy()

    def get_config(self) -> Dict[str, Any]:
        """현재 설정을 반환합니다."""
        return {
            'model': self.model,
            'generation_config': self.generation_config.copy(),
            'rpm_limit': self.rpm_limit
        }

    def update_generation_config(self, **kwargs):
        """생성 설정을 업데이트합니다."""
        for key, value in kwargs.items():
            if key == 'temperature' and not 0.0 <= value <= 1.0:
                raise ValueError("temperature는 0.0과 1.0 사이의 값이어야 합니다.")
            if key in ['max_output_tokens', 'top_k'] and value <= 0:
                raise ValueError(f"{key}는 양수여야 합니다.")
            if key == 'top_p' and not 0.0 <= value <= 1.0:
                raise ValueError("top_p는 0.0과 1.0 사이의 값이어야 합니다.")
            if key == 'thinking_budget':
                if value <= 0:
                    raise ValueError("thinking_budget는 양수여야 합니다.")
                # thinking_config 업데이트
                self.generation_config['thinking_config'] = types.ThinkingConfig(
                    thinking_budget=value
                )
                logger.info(f"thinking_budget가 {value}로 변경되었습니다.")
                continue
            
            self.generation_config[key] = value
            logger.info(f"{key}가 {value}로 변경되었습니다.")

    def set_thinking_budget(self, budget: int):
        """
        사고 과정 토큰 예산을 설정합니다.
        
        Args:
            budget (int): 새로운 thinking budget 값.
        
        Raises:
            ValueError: budget이 양수가 아닌 경우.
        """
        if budget <= 0:
            raise ValueError("thinking_budget는 양수여야 합니다.")
        
        self.generation_config['thinking_config'] = types.ThinkingConfig(
            thinking_budget=budget
        )
        self._init_args['thinking_budget'] = budget
        logger.info(f"Thinking budget이 {budget}으로 설정되었습니다.")

    def get_thinking_budget(self) -> int:
        """
        현재 설정된 thinking budget을 반환합니다.
        
        Returns:
            int: 현재 thinking budget 값.
        """
        thinking_config = self.generation_config.get('thinking_config')
        if thinking_config and hasattr(thinking_config, 'thinking_budget'):
            return thinking_config.thinking_budget
        return 8192  # 기본값 반환

    def set_response_schema(self, response_schema: Optional[Dict] = None,
                           response_mime_type: Optional[str] = None):
        """
        기본 응답 스키마를 설정합니다.
        
        Args:
            response_schema (dict, optional): JSON 응답 스키마.
            response_mime_type (str, optional): 응답 MIME 타입.
        """
        self.default_response_schema = response_schema
        
        if response_schema is not None:
            self.generation_config['response_schema'] = response_schema
            if response_mime_type is None:
                response_mime_type = "application/json"
            logger.info("기본 응답 스키마가 설정되었습니다.")
        else:
            self.generation_config.pop('response_schema', None)
            logger.info("기본 응답 스키마가 제거되었습니다.")
        
        if response_mime_type is not None:
            self.default_response_mime_type = response_mime_type
            self.generation_config['response_mime_type'] = response_mime_type
            logger.info(f"기본 응답 MIME 타입이 '{response_mime_type}'으로 설정되었습니다.")
        elif response_schema is None:
            # 스키마가 없고 MIME 타입도 지정하지 않았으면 제거
            self.generation_config.pop('response_mime_type', None)
            self.default_response_mime_type = None

if __name__ == "__main__":
    import json
    
    def test_response_schema():
        """response_schema 기능 테스트"""
        print("=== response_schema 테스트 시작 ===")
        
        test_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "language": {"type": "string"}
            },
            "required": ["message", "language"]
        }
        
        try:
            client = GeminiClient()
            response = client.send_message(
                "한국어로 '안녕하세요'라고 인사하고 JSON 형태로 응답해주세요. message와 language 필드를 포함해주세요.",
                response_schema=test_schema
            )
            
            print(f"응답: {response}")
            
            # JSON 파싱 테스트
            data = json.loads(response)
            print(f"파싱된 데이터: {data}")
            print(f"message: {data.get('message')}")
            print(f"language: {data.get('language')}")
            
        except Exception as e:
            print(f"테스트 실패: {e}")
        
        print()
    
    def test_stream_output():
        """스트림 출력 테스트"""
        print("=== 스트림 출력 테스트 시작 ===")
        
        try:
            client = GeminiClient()
            print("스트리밍 응답:")
            
            for chunk in client.send_message_stream("간단한 한국어 인사말을 해주세요."):
                print(chunk, end="", flush=True)
            
            print("\n=== 스트리밍 완료 ===")
            
        except Exception as e:
            print(f"스트리밍 테스트 실패: {e}")
        
        print()
    
    def test_stream_with_schema():
        """스키마가 적용된 스트림 출력 테스트"""
        print("=== 스키마 + 스트림 테스트 시작 ===")
        
        test_schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "mood": {"type": "string"}
            },
            "required": ["greeting", "mood"]
        }
        
        try:
            client = GeminiClient()
            print("JSON 스키마 스트리밍 응답:")
            
            full_response = ""
            for chunk in client.send_message_stream(
                "기분 좋은 인사말을 JSON 형태로 응답해주세요. greeting과 mood 필드를 포함해주세요.",
                response_schema=test_schema
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print(f"\n\n전체 응답: {full_response}")
            
            # JSON 파싱 테스트
            data = json.loads(full_response)
            print(f"파싱된 데이터: {data}")
            
        except Exception as e:
            print(f"스키마 스트리밍 테스트 실패: {e}")
    
    # 테스트 실행
    test_response_schema()
    test_stream_output()
    test_stream_with_schema()
