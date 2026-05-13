from datetime import datetime
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
