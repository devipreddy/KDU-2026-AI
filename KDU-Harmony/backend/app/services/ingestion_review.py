from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.document import Document
from app.models.enums import DocumentStatus, IngestionJobStatus
from app.models.ingestion_job import IngestionJob
from app.models.user import User
from app.services.clinical_metadata import (
    apply_clinical_metadata_to_document,
    extract_clinical_metadata,
)
from app.services.document_chunking import chunk_document_text
from app.services.embedding_pipeline import EmbeddingEncoder, index_document_chunks


class IngestionReviewError(ValueError):
    """Raised when a reviewed document cannot be approved for indexing."""


@dataclass(frozen=True)
class IngestionApprovalResult:
    document_id: uuid.UUID
    status: str
    chunk_count: int
    indexed_chunk_count: int
    embedding_dimension: int
    approved_at: datetime
    approved_by_user_id: uuid.UUID

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "status": self.status,
            "chunk_count": self.chunk_count,
            "indexed_chunk_count": self.indexed_chunk_count,
            "embedding_dimension": self.embedding_dimension,
            "approved_at": self.approved_at.isoformat(),
            "approved_by_user_id": str(self.approved_by_user_id),
        }


def approve_document_for_indexing(
    db: Session,
    *,
    document_id: uuid.UUID,
    approved_by: User,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
) -> IngestionApprovalResult:
    document = db.scalar(
        select(Document)
        .options(
            selectinload(Document.chunks),
            selectinload(Document.ingestion_jobs),
        )
        .where(Document.id == document_id)
    )
    if document is None:
        raise IngestionReviewError("Document was not found")

    if document.status == DocumentStatus.FAILED:
        raise IngestionReviewError("Failed documents must be reprocessed before approval")

    approved_at = datetime.now(UTC)
    latest_job = latest_ingestion_job(document)
    if latest_job is None:
        latest_job = IngestionJob(
            document=document,
            status=IngestionJobStatus.QUEUED,
            stage="approval_requested",
            queued_at=approved_at,
            job_metadata={"source": "manual_review_approval"},
        )
        db.add(latest_job)
        db.flush()

    existing_chunks = list(document.chunks)
    if should_rebuild_chunks_for_review(document, existing_chunks):
        text = read_processed_text_for_review(document)
        clinical_metadata = extract_clinical_metadata(text)
        apply_clinical_metadata_to_document(document, clinical_metadata)
        chunking = chunk_document_text(
            db,
            document=document,
            text=text,
            clinical_metadata=clinical_metadata,
            ocr_confidence=float(document.ocr_confidence) if document.ocr_confidence else None,
        )
        document.status = DocumentStatus.PROCESSED
        document.document_metadata = apply_review_gate_metadata(
            document.document_metadata,
            approved_at=approved_at,
            approved_by_user_id=approved_by.id,
            chunking=chunking.to_metadata(),
        )
        latest_job.status = IngestionJobStatus.RUNNING
        latest_job.stage = "approval_chunked"
        latest_job.error_message = None
        latest_job.job_metadata = {
            **(latest_job.job_metadata or {}),
            "review_gate": review_gate_metadata(
                approved_at=approved_at,
                approved_by_user_id=approved_by.id,
            ),
            "chunking": chunking.to_metadata(),
        }
        db.commit()
        chunk_count = chunking.total_chunk_count
    else:
        chunk_count = len(existing_chunks)
        document.document_metadata = apply_review_gate_metadata(
            document.document_metadata,
            approved_at=approved_at,
            approved_by_user_id=approved_by.id,
            chunking=None,
        )
        db.commit()

    indexing_result = index_document_chunks(
        db,
        document_id=document.id,
        ingestion_job=latest_job,
        retry_failed=True,
        encoder=encoder,
        collection=collection,
    )
    document = db.get(Document, document.id)
    if document is None:
        raise IngestionReviewError("Document disappeared after indexing")
    document.document_metadata = apply_review_gate_metadata(
        document.document_metadata,
        approved_at=approved_at,
        approved_by_user_id=approved_by.id,
        chunking=None,
        indexing=indexing_result.to_metadata(),
    )
    db.commit()

    return IngestionApprovalResult(
        document_id=document.id,
        status=document.status.value,
        chunk_count=chunk_count,
        indexed_chunk_count=indexing_result.indexed_chunk_count,
        embedding_dimension=indexing_result.embedding_dimension,
        approved_at=approved_at,
        approved_by_user_id=approved_by.id,
    )


def latest_ingestion_job(document: Document) -> IngestionJob | None:
    if not document.ingestion_jobs:
        return None
    return sorted(
        document.ingestion_jobs,
        key=lambda job: job.created_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )[0]


def read_processed_text_for_review(document: Document) -> str:
    extraction_metadata = (document.document_metadata or {}).get("extraction") or {}
    text_path = extraction_metadata.get("text_path")
    if not text_path:
        raise IngestionReviewError("Document does not have extracted text to approve")
    try:
        text = Path(text_path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise IngestionReviewError("Extracted text preview could not be read") from exc
    if not text:
        raise IngestionReviewError("Extracted text preview is empty")
    return text


def should_rebuild_chunks_for_review(document: Document, existing_chunks: list[Any]) -> bool:
    if not existing_chunks:
        return True
    metadata = document.document_metadata or {}
    for gate in current_review_gates(metadata):
        if gate.get("status") == "pending":
            return True
    for key in ("extraction", "ocr"):
        section = metadata.get(key)
        if not isinstance(section, dict):
            continue
        chunking = section.get("chunking")
        if isinstance(chunking, dict) and chunking.get("status") == "pending_review":
            return True
    return False


def current_review_gates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for key in ("extraction", "ocr", "review_gate"):
        section = metadata.get(key)
        if key == "review_gate":
            gate = section
        elif isinstance(section, dict):
            gate = section.get("review_gate")
        else:
            gate = None
        if isinstance(gate, dict):
            gates.append(gate)
    return gates


def apply_review_gate_metadata(
    metadata: dict[str, Any],
    *,
    approved_at: datetime,
    approved_by_user_id: uuid.UUID,
    chunking: dict[str, Any] | None,
    indexing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    review_gate = review_gate_metadata(
        approved_at=approved_at,
        approved_by_user_id=approved_by_user_id,
    )
    updated_metadata = {
        **(metadata or {}),
        "review_gate": review_gate,
    }
    for key in ("extraction", "ocr"):
        section = updated_metadata.get(key)
        if not isinstance(section, dict):
            continue
        updated_section = {
            **section,
            "review_gate": review_gate,
        }
        if chunking is not None:
            updated_section["chunking"] = chunking
        if indexing is not None:
            updated_section["indexing"] = indexing
        updated_metadata[key] = updated_section
    return updated_metadata


def review_gate_metadata(
    *,
    approved_at: datetime,
    approved_by_user_id: uuid.UUID,
) -> dict[str, str | bool]:
    return {
        "required": True,
        "status": "approved",
        "approved_at": approved_at.isoformat(),
        "approved_by_user_id": str(approved_by_user_id),
    }
