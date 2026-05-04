from __future__ import annotations

import re
from typing import Iterable

from app.core.models import ConversationMessage


class SecurityError(ValueError):
    pass


class SecurityGuard:
    _blocked_terms = {
        ".env",
        "openai_api_key",
        "secret",
        "token",
        "system prompt",
        "private key",
    }
    _secret_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"(?i)api[_ -]?key\s*[:=]\s*[A-Za-z0-9_-]{10,}"),
    ]

    def validate_user_text(self, text: str) -> str:
        sanitized = " ".join(text.strip().split())
        lowered = sanitized.lower()
        if any(term in lowered for term in self._blocked_terms):
            raise SecurityError(
                "The request references protected secrets or internal files and was blocked."
            )
        return sanitized

    def validate_messages(self, messages: Iterable[ConversationMessage]) -> None:
        for message in messages:
            self.ensure_no_secret_leak(message.text)

    def ensure_no_secret_leak(self, text: str) -> None:
        for pattern in self._secret_patterns:
            if pattern.search(text):
                raise SecurityError("Potential secret material detected in generated output.")
