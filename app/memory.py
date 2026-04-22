from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any


class SessionMemoryStore:
    def __init__(self, max_messages: int = 8) -> None:
        self.max_messages = max_messages
        self._lock = RLock()
        self._sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, user_query: str, assistant_response: str) -> None:
        with self._lock:
            messages = self._sessions[session_id]
            messages.extend(
                [
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response},
                ]
            )
            if len(messages) > self.max_messages:
                self._sessions[session_id] = messages[-self.max_messages :]
