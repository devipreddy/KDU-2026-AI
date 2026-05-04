from __future__ import annotations

from app.core.models import ConversationMessage


class ConversationPruner:
    def __init__(self, max_recent_messages: int, max_context_tokens: int, max_summary_chars: int) -> None:
        self.max_recent_messages = max_recent_messages
        self.max_context_tokens = max_context_tokens
        self.max_summary_chars = max_summary_chars

    def prune(
        self,
        summary: str,
        messages: list[ConversationMessage],
    ) -> tuple[str, list[ConversationMessage]]:
        if not messages:
            return summary, []

        recent = messages[-self.max_recent_messages :]
        removed = messages[: max(0, len(messages) - len(recent))]
        next_summary = self._merge_summary(summary, removed)

        while self._estimate_tokens(next_summary, recent) > self.max_context_tokens and len(recent) > 2:
            removed_chunk = recent[:2]
            recent = recent[2:]
            next_summary = self._merge_summary(next_summary, removed_chunk)

        return next_summary, recent

    def _merge_summary(self, summary: str, messages: list[ConversationMessage]) -> str:
        if not messages:
            return summary
        fragments = [summary.strip()] if summary.strip() else []
        for message in messages:
            compact = " ".join(message.text.strip().split())
            if len(compact) > 120:
                compact = compact[:117].rstrip() + "..."
            fragments.append(f"[{message.role}/{message.agent or 'system'}] {compact}")
        merged = " | ".join(fragment for fragment in fragments if fragment)
        if len(merged) > self.max_summary_chars:
            return merged[-self.max_summary_chars :]
        return merged

    def _estimate_tokens(self, summary: str, messages: list[ConversationMessage]) -> int:
        summary_tokens = len(summary) // 4
        message_tokens = sum(len(message.text) // 4 for message in messages)
        return summary_tokens + message_tokens
