"""키를 통해 Python 객체를 임시로 캐싱하기 위한 인메모리 저장소 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass
class MemoryStorage:
    """임의의 Python 객체를 위한 스레드 안전한 저장소."""

    _store: Dict[str, Any] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set(self, key: str, value: Any) -> None:
        """``key`` 아래에 ``value``를 저장합니다."""

        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """``key``에 해당하는 값을 검색하거나, 없을 경우 ``default``를 반환합니다."""

        with self._lock:
            return self._store.get(key, default)

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """``key``에 해당하는 값을 제거하고 반환합니다. 없을 경우 ``default``를 사용합니다."""

        with self._lock:
            return self._store.pop(key, default)

    def exists(self, key: str) -> bool:
        """``key``가 존재하면 ``True``를 반환합니다."""

        with self._lock:
            return key in self._store

    def keys(self) -> Iterable[str]:
        """현재 키들의 스냅샷을 반환합니다."""

        with self._lock:
            return tuple(self._store.keys())

    def clear(self) -> None:
        """모든 저장된 항목을 제거합니다."""

        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:  # pragma: no cover - delegating to exists
        return self.exists(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - convenience only
        return iter(self.keys())


default_storage = MemoryStorage()


def set_value(key: str, value: Any) -> None:
    """기본 저장소에 값을 저장합니다."""

    default_storage.set(key, value)


def get_value(key: str, default: Optional[Any] = None) -> Any:
    """기본 저장소에서 값을 가져옵니다."""

    return default_storage.get(key, default)


def remove_value(key: str, default: Optional[Any] = None) -> Any:
    """저장된 값을 제거하고 반환합니다."""

    return default_storage.pop(key, default)


def has_value(key: str) -> bool:
    """기본 저장소에 키가 존재하는지 확인합니다."""

    return default_storage.exists(key)


def clear_storage() -> None:
    """기본 저장소의 모든 값을 지웁니다."""

    default_storage.clear()

