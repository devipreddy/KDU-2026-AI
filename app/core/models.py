from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionPhase(StrEnum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


class ConversationMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    role: Literal["system", "user", "assistant", "summary"]
    text: str
    agent: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered: bool = True
    interrupted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageRecord(BaseModel):
    provider: str = "openai"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_hint_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriageDecision(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    target_agent: Literal["billing", "triage"]
    user_goal: str
    entities: dict[str, Any] = Field(default_factory=dict)
    routing_reason: str
    policy_flags: list[str] = Field(default_factory=list)


class HandoffState(BaseModel):
    state_version: int = 1
    trace_id: str
    session_id: str
    turn_id: str
    user_id: str
    source_agent: str
    target_agent: str
    current_intent: str
    intent_confidence: float
    entities: dict[str, Any] = Field(default_factory=dict)
    user_query: str
    conversation_summary: str = ""
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    retry_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerResult(BaseModel):
    agent: Literal["db", "vector"]
    success: bool
    latency_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    usage: UsageRecord | None = None


class ConsensusResult(BaseModel):
    answer_context: str
    confidence: float = Field(ge=0.0, le=1.0)
    winning_sources: list[str] = Field(default_factory=list)
    conflict_notes: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class BillingReply(BaseModel):
    reply_text: str
    short_summary: str
    follow_up_required: bool = False
    actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SessionMetrics(BaseModel):
    turns: int = 0
    interruptions: int = 0
    transcription_calls: int = 0
    tts_calls: int = 0
    llm_calls: int = 0
    tool_calls: int = 0


class SessionSnapshot(BaseModel):
    session_id: str
    user_id: str
    trace_id: str
    phase: SessionPhase
    current_agent: str = "triage"
    conversation_summary: str = ""
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    pending_handoff: HandoffState | None = None
    last_user_transcript: str = ""
    last_assistant_reply: str = ""
    metrics: SessionMetrics = Field(default_factory=SessionMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoggedEvent(BaseModel):
    trace_id: str
    session_id: str
    event_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentObservation(BaseModel):
    agent: str
    action: str
    detail: str
    latency_ms: int = 0
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextTurnRequest(BaseModel):
    user_id: str = "demo-user"
    transcript: str
    session_id: str | None = None
    trace_id: str | None = None
