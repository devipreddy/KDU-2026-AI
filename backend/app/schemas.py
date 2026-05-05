from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .context import ActorRole


class SessionResponse(BaseModel):
    client_secret: str
    expires_at: datetime
    user_id: str
    session_id: str
    role: ActorRole


class ThreadSummaryResponse(BaseModel):
    thread_id: str
    title: str | None = None
    conversation_mode: str = "ai"
    assigned_agent_name: str | None = None
    claimed_at: str | None = None
    last_message_preview: str | None = None
    last_updated_at: str | None = None


class ClaimHandoffRequest(BaseModel):
    display_name: str = Field(default="Travel Specialist")


class HumanMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ReleaseHandoffRequest(BaseModel):
    resume_ai: bool = True

