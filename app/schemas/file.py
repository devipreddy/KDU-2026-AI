from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel


class ApiUsageResponse(APIModel):
    id: int | None = None
    operation: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    response_ms: int | None = None
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ChunkResponse(APIModel):
    id: str
    chunk_index: int
    page_number: int | None = None
    token_count: int
    content: str
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class FileSummaryResponse(APIModel):
    id: str
    file_name: str
    file_type: str
    status: str
    title: str | None = None
    summary: str | None = None
    topic_tags: list[str] = Field(default_factory=list)
    chunk_count: int = 0
    created_at: datetime
    processed_at: datetime | None = None


class FileDetailResponse(APIModel):
    id: str
    file_name: str
    stored_path: str
    mime_type: str
    file_type: str
    size_bytes: int
    sha256: str
    status: str
    title: str | None = None
    description: str | None = None
    extracted_text: str | None = None
    cleaned_text: str | None = None
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    processing_error: str | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None
    chunks: list[ChunkResponse] = Field(default_factory=list)
    api_calls: list[ApiUsageResponse] = Field(default_factory=list)


class ProcessingJobResponse(APIModel):
    id: str
    file_id: str | None = None
    file_name: str
    sha256: str | None = None
    status: str
    progress_message: str | None = None
    error_message: str | None = None
    force_reprocess: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ProcessingJobStatusResponse(APIModel):
    job: ProcessingJobResponse
    file: FileDetailResponse | None = None


class FileProcessAcceptedResponse(APIModel):
    cached: bool
    job: ProcessingJobResponse
    file: FileDetailResponse | None = None
