from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class FactCheckDecision(BaseModel):
    verdict: Literal["approved", "needs_revision", "rejected"]
    route: Literal["write", "retry", "halt"]
    confidence: float = Field(ge=0.0, le=1.0)
    validated_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    corrections_required: list[str] = Field(default_factory=list)
    summary: str


class FlowStateModel(BaseModel):
    topic: str = ""
    research_output: str = ""
    fact_check: FactCheckDecision | None = None
    final_output: str = ""
    retry_count: int = 0
    max_retries: int = 2
    route_history: list[str] = Field(default_factory=list)
    last_error: str | None = None
    halt_reason: str | None = None


class TaskSnapshot(BaseModel):
    name: str
    description: str
    expected_output: str
    agent: str | None = None
    raw: str | None = None
    pydantic: dict[str, Any] | None = None


class RunArtifact(BaseModel):
    mode: str
    topic: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    final_output: str
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None

    def write_json(self, destination: Path) -> Path:
        destination.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return destination

