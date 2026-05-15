from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import require_roles
from app.core.config import settings
from app.db.session import get_db
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import (
    DocumentStatus,
    DocumentType,
    IngestionJobStatus,
    RoleName,
    SensitivityLevel,
)
from app.models.ingestion_job import IngestionJob
from app.models.user import User
from app.schemas.documents import (
    DocumentChunkResponse,
    DocumentDetailResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IngestionJobResponse,
)
from app.services.document_extraction import ExtractionError, process_ingestion_job
from app.services.document_ocr import QIANFAN_OCR_ENGINE, OcrError, process_ocr_ingestion_job
from app.services.document_storage import (
    UploadValidationError,
    sanitize_filename,
    store_encrypted_document,
)
from app.services.ingestion_review import IngestionReviewError, approve_document_for_indexing

router = APIRouter(prefix="/documents")

DbSession = Annotated[Session, Depends(get_db)]
UploadUser = Annotated[
    User,
    Depends(
        require_roles(
            RoleName.DOCTOR.value,
            RoleName.ADMIN.value,
            RoleName.RECORDS_STAFF.value,
        )
    ),
]
DocumentUser = UploadUser
ChunkCountMap = dict[uuid.UUID, tuple[int, int]]


def parse_json_object(raw_value: str | None, field_name: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} must be valid JSON",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} must be a JSON object",
        )
    return parsed


def parse_icd_codes(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw_value.split(",")]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="icd_codes must be a JSON string array or a comma-separated string",
        )
    return [item.strip() for item in parsed if item.strip()]


def enum_value(enum_class: type[DocumentType] | type[SensitivityLevel], value: str):
    try:
        return enum_class(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_class)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported value '{value}'. Allowed values: {allowed}",
        ) from exc


def default_document_type_for(mime_type: str) -> DocumentType:
    if mime_type == "application/pdf":
        return DocumentType.TYPED_PDF
    return DocumentType.CLINICAL_NOTE


def latest_job_for(document: Document) -> IngestionJob | None:
    if not document.ingestion_jobs:
        return None
    return sorted(document.ingestion_jobs, key=job_sort_value, reverse=True)[0]


def job_sort_value(job: IngestionJob) -> datetime:
    return job.created_at or datetime.min.replace(tzinfo=UTC)


def document_or_404(db: Session, document_id: uuid.UUID) -> Document:
    document = db.scalar(
        select(Document)
        .options(
            selectinload(Document.ingestion_jobs),
            selectinload(Document.chunks),
        )
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


def chunk_counts_for_documents(db: Session, document_ids: list[uuid.UUID]) -> ChunkCountMap:
    if not document_ids:
        return {}
    rows = db.execute(
        select(
            DocumentChunk.document_id,
            func.count(DocumentChunk.id),
            func.coalesce(
                func.sum(
                    case(
                        (DocumentChunk.indexing_status == "indexed", 1),
                        else_=0,
                    )
                ),
                0,
            ),
        )
        .where(DocumentChunk.document_id.in_(document_ids))
        .group_by(DocumentChunk.document_id)
    ).all()
    return {
        document_id: (int(chunk_count), int(indexed_count))
        for document_id, chunk_count, indexed_count in rows
    }


def latest_job_or_404(document: Document) -> IngestionJob:
    job = latest_job_for(document)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document has no ingestion job",
        )
    return job


def extraction_status(document: Document) -> str:
    metadata = document.document_metadata or {}
    extraction = metadata.get("extraction")
    if isinstance(extraction, dict) and extraction.get("text_path"):
        return "extracted"
    latest_job = latest_job_for(document)
    if latest_job is None:
        return "not_started"
    if latest_job.status == IngestionJobStatus.FAILED:
        return "failed"
    if latest_job.status == IngestionJobStatus.RUNNING:
        return latest_job.stage
    return "queued" if document.status == DocumentStatus.UPLOADED else document.status.value


def review_status(document: Document) -> str:
    metadata = document.document_metadata or {}
    for key in ("extraction", "ocr", "review_gate"):
        section = metadata.get(key)
        if isinstance(section, dict):
            gate = section if key == "review_gate" else section.get("review_gate")
            if isinstance(gate, dict) and gate.get("status"):
                return str(gate["status"])
    if document.status == DocumentStatus.REVIEW_REQUIRED:
        return "pending"
    if document.status == DocumentStatus.INDEXED:
        return "approved"
    return "not_required"


def read_extracted_text(document: Document, *, max_chars: int = 25000) -> tuple[str | None, int]:
    extraction = (document.document_metadata or {}).get("extraction") or {}
    if not isinstance(extraction, dict):
        return None, 0
    text_path = extraction.get("text_path")
    if not text_path:
        return None, 0
    try:
        text = Path(text_path).read_text(encoding="utf-8")
    except OSError:
        return None, 0
    return text[:max_chars], len(text)


def serialize_job(job: IngestionJob | None) -> IngestionJobResponse | None:
    if job is None:
        return None
    return IngestionJobResponse(
        id=job.id,
        status=job.status.value,
        stage=job.stage,
        attempts=job.attempts,
        error_message=job.error_message,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def chunk_type_for(chunk: DocumentChunk) -> str | None:
    metadata = chunk.retrieval_metadata or {}
    value = metadata.get("chunk_type")
    return str(value) if value else None


def serialize_chunk(chunk: DocumentChunk) -> DocumentChunkResponse:
    return DocumentChunkResponse(
        id=chunk.id,
        parent_chunk_id=chunk.parent_chunk_id,
        chunk_index=chunk.chunk_index,
        section=chunk.section,
        chunk_type=chunk_type_for(chunk),
        indexing_status=chunk.indexing_status,
        embedding_id=chunk.embedding_id,
        token_count=chunk.token_count,
        ocr_confidence=float(chunk.ocr_confidence) if chunk.ocr_confidence is not None else None,
        content_preview=chunk.content[:700],
    )


def serialize_document(
    document: Document,
    *,
    chunk_counts: ChunkCountMap | None = None,
) -> DocumentResponse:
    if chunk_counts is not None and document.id in chunk_counts:
        chunk_count, indexed_chunk_count = chunk_counts[document.id]
    elif chunk_counts is not None:
        chunk_count, indexed_chunk_count = 0, 0
    else:
        chunks = list(document.chunks)
        chunk_count = len(chunks)
        indexed_chunk_count = sum(1 for chunk in chunks if chunk.indexing_status == "indexed")
    latest_job = latest_job_for(document)
    return DocumentResponse(
        id=document.id,
        external_id=document.external_id,
        patient_ref=document.patient_ref,
        visit_id=document.visit_id,
        document_type=document.document_type.value,
        status=document.status.value,
        file_name=document.file_name,
        source_uri=document.source_uri,
        mime_type=document.mime_type,
        checksum_sha256=document.checksum_sha256,
        hospital=document.hospital,
        physician=document.physician,
        department=document.department,
        diagnosis=document.diagnosis,
        icd_codes=document.icd_codes,
        sensitivity_level=document.sensitivity_level.value,
        is_encrypted=document.is_encrypted,
        ocr_required=document.ocr_required,
        ocr_engine=document.ocr_engine,
        ocr_confidence=float(document.ocr_confidence)
        if document.ocr_confidence is not None
        else None,
        extraction_status=extraction_status(document),
        review_status=review_status(document),
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        latest_job=serialize_job(latest_job),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def serialize_document_detail(document: Document) -> DocumentDetailResponse:
    base = serialize_document(document).model_dump()
    extracted_text, extracted_text_char_count = read_extracted_text(document)
    jobs = sorted(document.ingestion_jobs, key=job_sort_value, reverse=True)
    chunks = sorted(document.chunks, key=lambda chunk: chunk.chunk_index)
    return DocumentDetailResponse(
        **base,
        extracted_text=extracted_text,
        extracted_text_char_count=extracted_text_char_count,
        metadata=document.document_metadata or {},
        jobs=[job for job in (serialize_job(job) for job in jobs) if job is not None],
        chunks=[serialize_chunk(chunk) for chunk in chunks[:60]],
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    current_user: DocumentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=2000)] = 100,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    source_filter: Annotated[str | None, Query(alias="source")] = None,
) -> list[DocumentResponse]:
    query_limit = limit if source_filter is None else max(limit, 2000)
    query = (
        select(Document)
        .options(selectinload(Document.ingestion_jobs))
        .order_by(Document.created_at.desc())
        .limit(query_limit)
    )
    if status_filter:
        try:
            query = query.where(Document.status == DocumentStatus(status_filter))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported document status",
            ) from exc
    documents = list(db.scalars(query).all())
    if source_filter:
        documents = [
            document
            for document in documents
            if (document.document_metadata or {}).get("upload_source") == source_filter
            or (document.document_metadata or {}).get("source") == source_filter
        ][:limit]
    chunk_counts = chunk_counts_for_documents(db, [document.id for document in documents])
    return [serialize_document(document, chunk_counts=chunk_counts) for document in documents]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_detail(
    document_id: uuid.UUID,
    current_user: DocumentUser,
    db: DbSession,
) -> DocumentDetailResponse:
    return serialize_document_detail(document_or_404(db, document_id))


@router.post("/{document_id}/extract", response_model=DocumentDetailResponse)
def extract_document_for_review(
    document_id: uuid.UUID,
    current_user: DocumentUser,
    db: DbSession,
) -> DocumentDetailResponse:
    document = document_or_404(db, document_id)
    if extraction_status(document) == "extracted" or document.status in {
        DocumentStatus.PROCESSED,
        DocumentStatus.INDEXED,
    }:
        return serialize_document_detail(document)
    job = latest_job_or_404(document)
    if job.status == IngestionJobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Extraction is already running for this document",
        )
    try:
        if document.ocr_required:
            process_ocr_ingestion_job(
                db,
                job.id,
                index_after_extraction=False,
                chunk_after_extraction=False,
            )
        else:
            process_ingestion_job(
                db,
                job.id,
                index_after_extraction=False,
                chunk_after_extraction=False,
            )
    except (ExtractionError, OcrError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return serialize_document_detail(document_or_404(db, document_id))


@router.post("/{document_id}/approve", response_model=DocumentDetailResponse)
def approve_document_review(
    document_id: uuid.UUID,
    current_user: DocumentUser,
    db: DbSession,
) -> DocumentDetailResponse:
    document = document_or_404(db, document_id)
    if document.status == DocumentStatus.INDEXED or review_status(document) == "approved":
        return serialize_document_detail(document)
    try:
        approve_document_for_indexing(db, document_id=document_id, approved_by=current_user)
    except IngestionReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return serialize_document_detail(document_or_404(db, document_id))


def duplicate_document_for_checksum(db: Session, checksum_sha256: str) -> Document | None:
    return db.scalar(
        select(Document)
        .options(
            selectinload(Document.ingestion_jobs),
            selectinload(Document.chunks),
        )
        .where(Document.checksum_sha256 == checksum_sha256)
        .order_by(Document.created_at.desc())
    )


def remove_duplicate_storage_object(storage_path: Path) -> None:
    try:
        resolved_path = storage_path.resolve()
        storage_root = settings.document_storage_root.resolve()
        if resolved_path.is_relative_to(storage_root):
            resolved_path.unlink(missing_ok=True)
    except OSError:
        return


def upload_response_for_document(
    document: Document,
    *,
    ingestion_job: IngestionJob,
    size_bytes: int,
    queued_at: datetime,
) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        id=document.id,
        external_id=document.external_id,
        patient_ref=document.patient_ref,
        visit_id=document.visit_id,
        document_type=document.document_type.value,
        status=document.status.value,
        file_name=document.file_name,
        source_uri=document.source_uri,
        mime_type=document.mime_type,
        checksum_sha256=document.checksum_sha256,
        size_bytes=size_bytes,
        is_encrypted=document.is_encrypted,
        ocr_required=document.ocr_required,
        ingestion_job_id=ingestion_job.id,
        ingestion_job_status=ingestion_job.status.value,
        queued_at=ingestion_job.queued_at or queued_at,
    )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    current_user: UploadUser,
    db: DbSession,
    file: Annotated[UploadFile, File()],
    patient_ref: Annotated[str, Form(min_length=1, max_length=80)],
    visit_id: Annotated[str | None, Form(max_length=80)] = None,
    document_type: Annotated[str | None, Form()] = None,
    hospital: Annotated[str | None, Form(max_length=160)] = None,
    physician: Annotated[str | None, Form(max_length=160)] = None,
    department: Annotated[str | None, Form(max_length=120)] = None,
    diagnosis: Annotated[str | None, Form(max_length=255)] = None,
    icd_codes: Annotated[str | None, Form()] = None,
    sensitivity_level: Annotated[str, Form()] = SensitivityLevel.MEDIUM.value,
    metadata: Annotated[str | None, Form()] = None,
) -> DocumentUploadResponse:
    content = await file.read()
    document_id = uuid.uuid4()

    try:
        stored = store_encrypted_document(
            document_id=document_id,
            original_filename=file.filename or "upload.bin",
            content_type=file.content_type,
            content=content,
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    duplicate = duplicate_document_for_checksum(db, stored.checksum_sha256)
    if duplicate is not None:
        remove_duplicate_storage_object(stored.storage_path)
        duplicate_job = latest_job_for(duplicate)
        if duplicate_job is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Duplicate document exists without an ingestion job",
            )
        existing_size = int(
            (duplicate.document_metadata or {}).get("size_bytes") or stored.size_bytes
        )
        return upload_response_for_document(
            duplicate,
            ingestion_job=duplicate_job,
            size_bytes=existing_size,
            queued_at=datetime.now(UTC),
        )

    selected_document_type = (
        enum_value(DocumentType, document_type)
        if document_type
        else default_document_type_for(stored.mime_type)
    )
    selected_sensitivity = enum_value(SensitivityLevel, sensitivity_level)
    metadata_payload = parse_json_object(metadata, "metadata")
    now = datetime.now(UTC)
    safe_filename = sanitize_filename(file.filename or "upload.bin")
    ocr_required = selected_document_type in {
        DocumentType.SCANNED_PDF,
        DocumentType.HANDWRITTEN_NOTE,
    }

    document = Document(
        id=document_id,
        external_id=f"DOC-UPLOAD-{document_id.hex[:16]}",
        patient_ref=patient_ref.strip(),
        visit_id=visit_id.strip() if visit_id else None,
        document_type=selected_document_type,
        status=DocumentStatus.UPLOADED,
        file_name=safe_filename,
        source_uri=stored.storage_uri,
        mime_type=stored.mime_type,
        checksum_sha256=stored.checksum_sha256,
        hospital=hospital.strip() if hospital else None,
        physician=physician.strip() if physician else None,
        department=department.strip() if department else None,
        diagnosis=diagnosis.strip() if diagnosis else None,
        icd_codes=parse_icd_codes(icd_codes),
        sensitivity_level=selected_sensitivity,
        is_encrypted=True,
        ocr_required=ocr_required,
        ocr_engine=QIANFAN_OCR_ENGINE if ocr_required else None,
        created_by=current_user,
        document_metadata={
            **metadata_payload,
            "original_filename": file.filename,
            "uploaded_by_user_id": str(current_user.id),
            "size_bytes": stored.size_bytes,
            "encrypted_checksum_sha256": stored.encrypted_checksum_sha256,
            "encryption_version": stored.encryption_version,
            "storage_path": str(stored.storage_path),
            "validation": {
                "mime_type": stored.mime_type,
                "checksum_algorithm": "sha256",
            },
        },
    )
    ingestion_job = IngestionJob(
        document=document,
        status=IngestionJobStatus.QUEUED,
        stage="uploaded",
        attempts=0,
        queued_at=now,
        job_metadata={
            "source": "secure_upload",
            "storage_uri": stored.storage_uri,
            "ocr_required": ocr_required,
        },
    )

    db.add(document)
    db.add(ingestion_job)
    db.commit()
    db.refresh(document)
    db.refresh(ingestion_job)

    return upload_response_for_document(
        document,
        ingestion_job=ingestion_job,
        size_bytes=stored.size_bytes,
        queued_at=now,
    )
