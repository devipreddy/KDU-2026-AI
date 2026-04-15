from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    session_id: str
    query: str
    rewritten_query: str
    query_needs_rewrite: bool
    chat_history: list[dict[str, str]]
    conversation_summary: str
    retrieved_docs: list[dict[str, Any]]
    deduplicated_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    selected_context: list[dict[str, Any]]
    answer: str
    confidence: float
    iteration_count: int
    metadata: dict[str, Any]
