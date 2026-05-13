from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.models.document import Document
from app.models.enums import (
    DocumentStatus,
    DocumentType,
    IngestionJobStatus,
    RoleName,
    SensitivityLevel,
)
from app.models.ingestion_job import IngestionJob
from app.models.user import User
from app.schemas.documents import DocumentUploadResponse
from app.services.document_storage import (
    UploadValidationError,
    sanitize_filename,
    store_encrypted_document,
)

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
        ocr_engine="paddleocr" if ocr_required else None,
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
        size_bytes=stored.size_bytes,
        is_encrypted=document.is_encrypted,
        ocr_required=document.ocr_required,
        ingestion_job_id=ingestion_job.id,
        ingestion_job_status=ingestion_job.status.value,
        queued_at=ingestion_job.queued_at or now,
    )
