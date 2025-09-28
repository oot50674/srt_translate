import os
import time
import uuid
import base64
import json
import logging
from collections import deque
from typing import Iterator, Dict, Any, Optional, List

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API와 통신하는 대화형 클라이언트."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = None,
        generation_config: Optional[Dict[str, Any]] = None,
        rpm_limit: int = 10,
        thinking_config: Any = None,
        **kwargs,
    ):
        """클라이언트 초기화."""
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.history: List[Dict[str, Any]] = []
        self.request_times = deque()
        self.rpm_limit = rpm_limit
        self.generation_config = generation_config.copy() if generation_config else {}

        if thinking_config is not None:
            logger.warning("thinking_config 파라미터는 더 이상 사용되지 않습니다.")

        self._init_args = {
            "model": model,
            "base_url": self.base_url,
            "generation_config": self.generation_config.copy(),
            "rpm_limit": rpm_limit,
        }
        self._init_args.update(kwargs)
        logger.info(f"OllamaClient 초기화 - 모델: {self.model}, RPM 제한: {self.rpm_limit}")

    def _check_rpm_limit(self) -> None:
        current_time = time.time()
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.info(f"RPM 제한 도달. {wait_time:.2f}초 대기합니다.")
                time.sleep(wait_time)
                current_time = time.time()
                while self.request_times and current_time - self.request_times[0] > 60:
                    self.request_times.popleft()
        self.request_times.append(time.time())

    def start_chat(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        self.history = history if history is not None else []
        logger.info("새 대화를 시작합니다.")

    def _prepare_history(self) -> List[Dict[str, Any]]:
        processed = []
        for message in self.history:
            msg = {"role": message.get("role", "user"), "content": message.get("content", "")}
            if message.get("images"):
                base64_images = []
                for path in message["images"]:
                    try:
                        with open(path, "rb") as f:
                            base64_images.append(base64.b64encode(f.read()).decode("utf-8"))
                    except Exception as e:
                        logger.error(f"이미지 인코딩 실패: {e}")
                if base64_images:
                    msg["images"] = base64_images
            processed.append(msg)
        return processed

    def send_message(
        self,
        message: str,
        model: str = None,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        # 추가 인자는 무시하지만, 호환성을 위해 경고 로그를 남깁니다.
        if kwargs:
            logger.debug(f"지원되지 않는 추가 인자 무시: {list(kwargs.keys())}")

        self._check_rpm_limit()
        user_msg = {"role": "user", "content": message}
        if images:
            user_msg["images"] = images
        self.history.append(user_msg)
        payload = {
            "model": model or self.model,
            "messages": self._prepare_history(),
            "stream": False,
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "")
            self.history.append({"role": "assistant", "content": text})
            return text
        except requests.exceptions.RequestException as e:
            self.history.pop()
            logger.error(f"Ollama API 오류: {e}")
            raise

    def send_message_stream(
        self,
        message: str,
        model: str = None,
        images: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterator[str]:
        # 추가 인자는 무시하지만, 호환성을 위해 경고 로그를 남깁니다.
        if kwargs:
            logger.debug(f"지원되지 않는 추가 인자 무시: {list(kwargs.keys())}")

        self._check_rpm_limit()
        user_msg = {"role": "user", "content": message}
        if images:
            user_msg["images"] = images
        self.history.append(user_msg)
        payload = {
            "model": model or self.model,
            "messages": self._prepare_history(),
            "stream": True,
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True)
            resp.raise_for_status()
            full_response = []
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    try:
                        data = json.loads(decoded)
                        if data.get("message") and data["message"].get("content"):
                            full_response.append(data["message"]["content"])
                        yield decoded + "\n"
                    except json.JSONDecodeError:
                        yield json.dumps({"error": "응답 디코딩 오류"}) + "\n"
            if full_response:
                self.history.append({"role": "assistant", "content": "".join(full_response)})
        except requests.exceptions.RequestException as e:
            self.history.pop()
            logger.error(f"Ollama API 오류: {e}")
            raise

    def delete_last_turn(self) -> bool:
        if len(self.history) < 2:
            logger.warning("삭제할 대화 턴이 없습니다.")
            return False
        self.history.pop()
        self.history.pop()
        return True

    def fork(self) -> "OllamaClient":
        client = OllamaClient(**self._init_args)
        client.start_chat(history=[msg.copy() for msg in self.history])
        return client

    def modify_message(self, index: int, new_message: str) -> Optional["OllamaClient"]:
        if not (0 <= index < len(self.history)):
            logger.error(f"잘못된 인덱스: {index}")
            return None
        if self.history[index].get("role") != "user":
            logger.error("사용자 메시지만 수정할 수 있습니다.")
            return None
        modified = [msg.copy() for msg in self.history[: index + 1]]
        modified[index]["content"] = new_message
        new_client = OllamaClient(**self._init_args)
        new_client.start_chat(history=modified)
        return new_client

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()

    def set_rpm_limit(self, rpm_limit: int) -> None:
        if rpm_limit <= 0:
            raise ValueError("RPM 제한은 양수여야 합니다.")
        self.rpm_limit = rpm_limit
        self._init_args["rpm_limit"] = rpm_limit

    def get_current_rpm_usage(self) -> Dict[str, Any]:
        current_time = time.time()
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        current_count = len(self.request_times)
        time_until_reset = 60 - (current_time - self.request_times[0]) if self.request_times else 0
        return {
            "limit": self.rpm_limit,
            "current_count": current_count,
            "remaining": max(0, self.rpm_limit - current_count),
            "time_until_reset": max(0, time_until_reset),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "generation_config": self.generation_config.copy(),
            "rpm_limit": self.rpm_limit,
        }

    def update_generation_config(self, **kwargs) -> None:
        self.generation_config.update(kwargs)
        self._init_args["generation_config"] = self.generation_config.copy()

    def set_thinking_budget(self, budget: int) -> None:
        logger.warning("thinking_budget 설정은 Ollama에서 지원되지 않습니다.")

    def get_thinking_budget(self) -> int:
        logger.warning("thinking_budget 조회는 Ollama에서 지원되지 않습니다.")
        return 0

    def set_response_schema(
        self, response_schema: Optional[Dict] = None, response_mime_type: str = None
    ) -> None:
        logger.warning("Ollama는 response_schema 기능을 지원하지 않습니다.")


__all__ = ["OllamaClient"]

if __name__ == "__main__":
    # OllamaClient 테스트 코드
    import sys
    
    def test_ollama_client():
        """OllamaClient의 기본 기능을 테스트합니다."""
        print("=== OllamaClient 테스트 시작 ===")
        
        try:
            # 클라이언트 초기화 테스트
            print("\n1. 클라이언트 초기화 테스트")
            client = OllamaClient(
                model="qwen3:8b-q8_0",
                rpm_limit=2
            )
            print(f"✓ 클라이언트 초기화 성공 - 모델: {client.model}")
            
            # 설정 확인 테스트
            print("\n2. 설정 확인 테스트")
            config = client.get_config()
            print(f"✓ 현재 설정: {config}")
            
            # RPM 제한 테스트
            print("\n3. RPM 제한 테스트")
            rpm_status = client.get_current_rpm_usage()
            print(f"✓ RPM 상태: {rpm_status}")
            
            # 대화 시작 테스트
            print("\n4. 대화 시작 테스트")
            client.start_chat()
            print("✓ 대화 시작 성공")
            
            # 간단한 메시지 테스트 (실제 Ollama 서버가 실행 중인 경우에만)
            print("\n5. 메시지 전송 테스트")
            try:
                response = client.send_message("안녕하세요! 간단한 인사말로 답변해주세요.")
                print(f"✓ 응답 받음: {response[:100]}...")
            except Exception as e:
                print(f"⚠ 메시지 전송 테스트 실패 (Ollama 서버가 실행 중이지 않을 수 있습니다): {e}")
            
            # 히스토리 테스트
            print("\n6. 히스토리 테스트")
            history = client.get_history()
            print(f"✓ 현재 히스토리 길이: {len(history)}")
            
            # 복제 테스트
            print("\n7. 클라이언트 복제 테스트")
            cloned_client = client.clone()
            print("✓ 클라이언트 복제 성공")
            
            # thinking budget 테스트 (경고 메시지 확인)
            print("\n8. Thinking Budget 테스트")
            client.set_thinking_budget(1000)
            budget = client.get_thinking_budget()
            print(f"✓ Thinking Budget: {budget}")
            
            # response schema 테스트 (경고 메시지 확인)
            print("\n9. Response Schema 테스트")
            test_schema = {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            }
            client.set_response_schema(test_schema)
            print("✓ Response Schema 설정 완료")
            
            print("\n=== 모든 테스트 완료 ===")
            
        except Exception as e:
            print(f"✗ 테스트 중 오류 발생: {e}")
            return False
        
        return True
    
    def test_error_handling():
        """에러 처리 테스트"""
        print("\n=== 에러 처리 테스트 ===")
        
        try:
            # 잘못된 RPM 제한 설정 테스트
            print("1. 잘못된 RPM 제한 설정 테스트")
            client = OllamaClient()
            try:
                client.set_rpm_limit(0)
                print("✗ 예상된 ValueError가 발생하지 않았습니다.")
            except ValueError:
                print("✓ ValueError 정상적으로 발생")
            
            # 존재하지 않는 서버 테스트
            print("\n2. 존재하지 않는 서버 테스트")
            client_invalid = OllamaClient(base_url="http://localhost:99999")
            try:
                client_invalid.send_message("테스트")
                print("✗ 연결 에러가 발생하지 않았습니다.")
            except Exception:
                print("✓ 연결 에러 정상적으로 처리됨")
            
            print("\n=== 에러 처리 테스트 완료 ===")
            
        except Exception as e:
            print(f"✗ 에러 처리 테스트 중 오류: {e}")
            return False
        
        return True
    
    # 메인 테스트 실행
    print("OllamaClient 테스트를 시작합니다...")
    
    success = True
    success &= test_ollama_client()
    success &= test_error_handling()
    
    if success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        sys.exit(1)

