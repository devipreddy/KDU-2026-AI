from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: UUID
    external_id: str
    patient_ref: str
    visit_id: str | None
    document_type: str
    status: str
    file_name: str
    source_uri: str
    mime_type: str
    checksum_sha256: str
    size_bytes: int
    is_encrypted: bool
    ocr_required: bool
    ingestion_job_id: UUID
    ingestion_job_status: str
    queued_at: datetime


class IngestionJobResponse(BaseModel):
    id: UUID
    status: str
    stage: str
    attempts: int
    error_message: str | None
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None


class DocumentChunkResponse(BaseModel):
    id: UUID
    parent_chunk_id: UUID | None
    chunk_index: int
    section: str | None
    chunk_type: str | None
    indexing_status: str
    embedding_id: str | None
    token_count: int | None
    ocr_confidence: float | None
    content_preview: str


class DocumentResponse(BaseModel):
    id: UUID
    external_id: str
    patient_ref: str
    visit_id: str | None
    document_type: str
    status: str
    file_name: str
    source_uri: str
    mime_type: str
    checksum_sha256: str
    hospital: str | None
    physician: str | None
    department: str | None
    diagnosis: str | None
    icd_codes: list[str]
    sensitivity_level: str
    is_encrypted: bool
    ocr_required: bool
    ocr_engine: str | None
    ocr_confidence: float | None
    extraction_status: str
    review_status: str
    chunk_count: int
    indexed_chunk_count: int
    latest_job: IngestionJobResponse | None
    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(DocumentResponse):
    extracted_text: str | None
    extracted_text_char_count: int
    metadata: dict[str, Any]
    jobs: list[IngestionJobResponse]
    chunks: list[DocumentChunkResponse]
