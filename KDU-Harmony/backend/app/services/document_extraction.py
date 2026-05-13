from __future__ import annotations

import argparse
import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import fitz
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus, DocumentType, IngestionJobStatus
from app.models.ingestion_job import IngestionJob
from app.services.document_storage import StorageReadError, read_encrypted_document


class ExtractionError(ValueError):
    """Raised when a document cannot be extracted by the typed extraction pipeline."""


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    extractor: str
    page_count: int | None
    non_empty_page_count: int | None

    @property
    def char_count(self) -> int:
        return len(self.text)


def extract_plain_text(content: bytes) -> ExtractionResult:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractionError("Plain text document is not valid UTF-8") from exc

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ExtractionError("Plain text document did not contain extractable text")

    return ExtractionResult(
        text=normalized,
        extractor="plain_text",
        page_count=None,
        non_empty_page_count=None,
    )


def extract_typed_pdf(content: bytes) -> ExtractionResult:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            page_texts = [page.get_text("text").strip() for page in pdf]
    except Exception as exc:
        raise ExtractionError("Typed PDF text extraction failed") from exc

    text = "\n\n".join(page_text for page_text in page_texts if page_text).strip()
    if not text:
        raise ExtractionError("Typed PDF did not contain extractable text")

    return ExtractionResult(
        text=text,
        extractor="pymupdf",
        page_count=len(page_texts),
        non_empty_page_count=sum(1 for page_text in page_texts if page_text),
    )


def extract_document_text(document: Document, content: bytes) -> ExtractionResult:
    if document.ocr_required or document.document_type in {
        DocumentType.SCANNED_PDF,
        DocumentType.HANDWRITTEN_NOTE,
    }:
        raise ExtractionError("Document requires OCR and is not supported by typed extraction")

    if document.mime_type == "text/plain":
        return extract_plain_text(content)

    if document.mime_type == "application/pdf" and document.document_type == DocumentType.TYPED_PDF:
        return extract_typed_pdf(content)

    raise ExtractionError(
        "Document MIME type or document type is not supported by typed extraction"
    )


def processed_text_path_for(document_id: uuid.UUID) -> Path:
    return settings.processed_text_root.resolve() / f"{document_id}.txt"


def write_processed_text(document_id: uuid.UUID, text: str) -> Path:
    output_path = processed_text_path_for(document_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return output_path


def _storage_path_for(document: Document) -> Path:
    raw_storage_path = document.document_metadata.get("storage_path")
    if not raw_storage_path:
        raise ExtractionError("Document metadata is missing storage_path")
    return Path(raw_storage_path)


def process_ingestion_job(db: Session, job_id: uuid.UUID) -> ExtractionResult:
    job = db.scalar(
        select(IngestionJob)
        .options(selectinload(IngestionJob.document))
        .where(IngestionJob.id == job_id)
    )
    if job is None:
        raise ExtractionError("Ingestion job was not found")

    document = job.document
    started_at = datetime.now(UTC)
    job.status = IngestionJobStatus.RUNNING
    job.stage = "extracting_text"
    job.attempts += 1
    job.started_at = started_at
    job.error_message = None
    document.status = DocumentStatus.EXTRACTING
    db.commit()

    try:
        content = read_encrypted_document(
            storage_path=_storage_path_for(document),
            document_id=document.id,
        )
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != document.checksum_sha256:
            raise ExtractionError("Stored document checksum does not match metadata")

        result = extract_document_text(document, content)
        processed_path = write_processed_text(document.id, result.text)
        extracted_at = datetime.now(UTC)

        document.document_metadata = {
            **document.document_metadata,
            "extraction": {
                "extractor": result.extractor,
                "extracted_at": extracted_at.isoformat(),
                "text_path": str(processed_path),
                "text_uri": f"local-processed://{processed_path.name}",
                "char_count": result.char_count,
                "page_count": result.page_count,
                "non_empty_page_count": result.non_empty_page_count,
                "checksum_verified": True,
            },
        }
        document.status = DocumentStatus.PROCESSED
        job.status = IngestionJobStatus.SUCCEEDED
        job.stage = "text_extracted"
        job.finished_at = extracted_at
        job.job_metadata = {
            **job.job_metadata,
            "extraction": document.document_metadata["extraction"],
        }
        db.commit()
        return result
    except (ExtractionError, StorageReadError) as exc:
        failed_at = datetime.now(UTC)
        document.status = DocumentStatus.FAILED
        job.status = IngestionJobStatus.FAILED
        job.stage = "text_extraction_failed"
        job.error_message = str(exc)
        job.finished_at = failed_at
        job.job_metadata = {
            **job.job_metadata,
            "failed_at": failed_at.isoformat(),
        }
        db.commit()
        raise ExtractionError(str(exc)) from exc


def process_queued_ingestion_jobs(db: Session, *, limit: int = 25) -> list[uuid.UUID]:
    jobs = db.scalars(
        select(IngestionJob)
        .join(IngestionJob.document)
        .where(IngestionJob.status == IngestionJobStatus.QUEUED)
        .where(Document.ocr_required.is_(False))
        .order_by(IngestionJob.created_at)
        .limit(limit)
    ).all()

    processed_job_ids: list[uuid.UUID] = []
    for job in jobs:
        process_ingestion_job(db, job.id)
        processed_job_ids.append(job.id)
    return processed_job_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text for queued typed ingestion jobs.")
    parser.add_argument("--job-id", type=uuid.UUID)
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        if args.job_id:
            process_ingestion_job(db, args.job_id)
            print(f"Processed ingestion job {args.job_id}")
            return

        processed_job_ids = process_queued_ingestion_jobs(db, limit=args.limit)
        print(f"Processed {len(processed_job_ids)} ingestion jobs")
