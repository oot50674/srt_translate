"""
별도 프로세스에서 faster-whisper를 실행해 언로드 시 크래시를 피하는 헬퍼.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Optional


class WhisperEngineInitError(RuntimeError):
    """WhisperEngine 초기화 실패 시 사용되는 예외."""


def _worker(req_q: "mp.Queue[Dict[str, Any] | None]", res_q: "mp.Queue[Dict[str, Any]]", engine_kwargs: Dict[str, Any]) -> None:
    """WhisperEngine을 별도 프로세스에서 구동합니다."""
    try:
        from module.WhisperEngine import WhisperEngine

        engine = WhisperEngine(**engine_kwargs)
    except Exception as exc:  # pragma: no cover - 방어적 초기화
        res_q.put({"ok": False, "error": str(exc), "init_error": True})
        return

    while True:
        task = req_q.get()
        if task is None:
            break
        try:
            audio_path = task.get("audio_path")
            kwargs = task.get("kwargs") or {}
            segments_gen, info = engine.transcribe(audio_path, **kwargs)

            segments_data = []
            texts = []
            for seg in segments_gen:
                start_value = float(getattr(seg, "start", 0.0))
                end_value = float(getattr(seg, "end", start_value))
                text_value = str(getattr(seg, "text", "")).strip()
                if text_value:
                    texts.append(text_value)
                segments_data.append(
                    {
                        "start": start_value,
                        "end": end_value,
                        "text": text_value,
                    }
                )

            info_data = {
                "language": getattr(info, "language", None) if info else None,
                "language_probability": getattr(info, "language_probability", None) if info else None,
            }

            res_q.put(
                {
                    "ok": True,
                    "segments": segments_data,
                    "text": " ".join(texts).strip(),
                    "info": info_data,
                }
            )
        except Exception as exc:  # pragma: no cover - 런타임 방어
            res_q.put({"ok": False, "error": str(exc)})

    try:
        del engine
    except Exception:
        pass


class WhisperProcessRunner:
    """WhisperEngine을 별도 프로세스에서 실행하는 래퍼."""

    def __init__(self, **engine_kwargs: Any) -> None:
        self.engine_kwargs = engine_kwargs
        # CUDA 초기화 오류를 피하기 위해 spawn 컨텍스트를 강제한다.
        self._ctx = mp.get_context("spawn")
        self.req_q: mp.Queue = self._ctx.Queue()
        self.res_q: mp.Queue = self._ctx.Queue()
        self.proc: Optional[mp.Process] = None

    def _ensure_running(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.proc = self._ctx.Process(
            target=_worker,
            args=(self.req_q, self.res_q, self.engine_kwargs),
            daemon=True,
        )
        self.proc.start()

    def transcribe(self, audio_path: str, **kwargs: Any) -> Dict[str, Any]:
        self._ensure_running()
        self.req_q.put({"audio_path": audio_path, "kwargs": kwargs})
        result = self.res_q.get()
        if not result.get("ok"):
            if result.get("init_error"):
                self.stop()
                error = result.get("error", "WhisperEngine 초기화 실패")
                raise WhisperEngineInitError(error)
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Whisper 프로세스 오류: {error}")
        return result

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc is None:
            return
        try:
            self.req_q.put(None)
            self.proc.join(timeout=timeout)
            if self.proc.is_alive():
                self.proc.terminate()
        finally:
            self.proc = None
