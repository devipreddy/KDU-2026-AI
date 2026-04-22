from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class CostBreakdown(BaseModel):
    amount: float = 0.0
    currency: str = "USD"


class ChatResponse(BaseModel):
    request_id: str
    response: str
    route: Literal["tool", "llm", "hybrid", "direct"]
    tool_used: list[str] = Field(default_factory=list)
    model_used: str
    tokens: TokenUsage
    cost: CostBreakdown
    latency_ms: int
    cache_hit: bool = False
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    uptime: str
    dependencies: dict[str, str]


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    llm_calls: int
    tool_calls: dict[str, int]
    cache_hit_rate: float
    token_usage: dict[str, int]


class CacheInvalidateRequest(BaseModel):
    key: str = Field(..., min_length=1)


class AssistantState(TypedDict, total=False):
    request_id: str
    session_id: str
    user_id: str | None
    user_query: str
    cleaned_query: str
    intent: str | None
    detected_domains: list[str]
    route: Literal["tool", "llm", "hybrid", "direct"]
    tool_name: str | None
    tool_names: list[str]
    tool_input: dict[str, Any] | None
    tool_output: dict[str, Any] | list[dict[str, Any]] | None
    normalized_tool_output: dict[str, Any] | list[dict[str, Any]] | None
    llm_model: str
    llm_response: str | None
    complexity: Literal["low", "high"] | None
    confidence: float | None
    error: str | None
    retry_count: int
    cache_hit: bool
    memory_context: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    planner_messages: list[dict[str, Any]]
    tokens_used: dict[str, int]
    llm_calls: int
    fallback_mode: str | None
    planner_error: str | None
    summary_error: str | None
    latency_ms: int
    cache_key: str | None
    started_at: datetime
    completed_at: datetime | None


def build_initial_state(payload: ChatRequest) -> AssistantState:
    return AssistantState(
        request_id=str(uuid4()),
        session_id=payload.session_id or str(uuid4()),
        user_id=payload.user_id,
        user_query=payload.query,
        cleaned_query=payload.query,
        intent=None,
        detected_domains=[],
        route="llm",
        tool_name=None,
        tool_names=[],
        tool_input=None,
        tool_output=None,
        normalized_tool_output=None,
        llm_model="",
        llm_response=None,
        complexity=None,
        confidence=None,
        error=None,
        retry_count=0,
        cache_hit=False,
        memory_context=[],
        messages=[],
        planner_messages=[],
        tokens_used={"input": 0, "output": 0, "total": 0},
        llm_calls=0,
        fallback_mode=None,
        planner_error=None,
        summary_error=None,
        latency_ms=0,
        cache_key=None,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
    )
