from __future__ import annotations

from dataclasses import dataclass, field

from .constants import DEFAULT_SUMMARY_FALLBACK
from .utils import read_json_file, write_json_file


@dataclass
class SessionMemory:
    chat_history: list[dict[str, str]] = field(default_factory=list)
    summary: str = DEFAULT_SUMMARY_FALLBACK


class SessionMemoryManager:
    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._sessions: dict[str, SessionMemory] = {}
        self._load_store()

    def _load_store(self) -> None:
        raw = read_json_file(self.store_path, default={})
        self._sessions = {
            session_id: SessionMemory(
                chat_history=value.get("chat_history", []),
                summary=value.get("summary", DEFAULT_SUMMARY_FALLBACK),
            )
            for session_id, value in raw.items()
        }

    def _persist(self) -> None:
        payload = {
            session_id: {
                "chat_history": session.chat_history,
                "summary": session.summary,
            }
            for session_id, session in self._sessions.items()
        }
        write_json_file(self.store_path, payload)

    def load(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
            self._persist()
        return self._sessions[session_id]

    def update(
        self,
        session_id: str,
        query: str,
        answer: str,
        summary: str | None = None,
    ) -> SessionMemory:
        session = self.load(session_id)
        session.chat_history.append({"role": "user", "content": query})
        session.chat_history.append({"role": "assistant", "content": answer})
        if summary:
            session.summary = summary
        self._persist()
        return session

    def export_session(self, session_id: str) -> dict[str, object]:
        session = self.load(session_id)
        history = []
        pending_query = None
        for item in session.chat_history:
            if item["role"] == "user":
                pending_query = item["content"]
            elif item["role"] == "assistant" and pending_query is not None:
                history.append({"query": pending_query, "response": item["content"]})
                pending_query = None
        return {
            "session_id": session_id,
            "history": history,
            "summary": session.summary,
        }

    def list_sessions(self) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        for session_id, session in self._sessions.items():
            turns = sum(1 for item in session.chat_history if item["role"] == "user")
            last_user_message = next(
                (item["content"] for item in reversed(session.chat_history) if item["role"] == "user"),
                "",
            )
            sessions.append(
                {
                    "session_id": session_id,
                    "turns": turns,
                    "summary": session.summary,
                    "last_user_message": last_user_message,
                }
            )
        return sorted(sessions, key=lambda item: (item["turns"], item["session_id"]), reverse=True)
