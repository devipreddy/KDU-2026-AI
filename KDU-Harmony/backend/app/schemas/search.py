from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=25)
    candidate_limit: int = Field(default=20, ge=1, le=250)
    rerank_top_n: int = Field(default=8, ge=0, le=100)
    include_llm_answer: bool = True
    require_llm_answer: bool = False
    collection: str | None = None


class SearchUserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    roles: list[str]


class SearchAnswerResponse(BaseModel):
    status: str
    answer: str
    provider: str | None
    model: str | None
    citations: list[dict[str, Any]]
    latency_ms: float | None = None
    error: str | None = None


class SearchHitResponse(BaseModel):
    final_rank: int
    patient_display_ref: str | None
    matched_chunk: dict[str, Any]
    parent_context: dict[str, Any] | None
    citation: dict[str, Any]
    confidence: dict[str, Any]
    retrieval: dict[str, Any]
    redactions: list[str]


class SearchResponse(BaseModel):
    query: str
    user: SearchUserResponse
    hit_count: int
    answer: SearchAnswerResponse
    hits: list[SearchHitResponse]
    timeline: list[dict[str, Any]]
    pipeline: dict[str, Any]
