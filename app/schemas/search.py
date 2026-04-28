from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import APIModel


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    file_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class SearchHitResponse(APIModel):
    chunk_id: str
    file_id: str
    file_name: str
    page_number: int | None = None
    score: float
    content: str
    context: str
    metadata: dict[str, str | int | float | list[int] | None] = Field(default_factory=dict)


class SearchResponse(APIModel):
    query: str
    hits: list[SearchHitResponse]
