from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCacheStore:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._items: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if not entry:
                return None
            if entry.expires_at <= now:
                self._items.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = CacheEntry(
                value=value,
                expires_at=time.time() + self.ttl_seconds,
            )

    def invalidate(self, key: str) -> bool:
        with self._lock:
            return self._items.pop(key, None) is not None
