from __future__ import annotations

import argparse
import hashlib
import io
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus, DocumentType, IngestionJobStatus
from app.models.ingestion_job import IngestionJob
from app.services.document_extraction import ExtractionError, write_processed_text
from app.services.document_storage import StorageReadError, read_encrypted_document


class OcrError(ValueError):
    """Raised when OCR cannot extract text from an OCR-required document."""


@dataclass(frozen=True)
class OcrResult:
    text: str
    engine: str
    page_count: int
    non_empty_page_count: int
    confidence: float
    review_required: bool

    @property
    def char_count(self) -> int:
        return len(self.text)


def configure_tesseract() -> None:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _page_images_from_pdf(content: bytes) -> list[Image.Image]:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            scale = settings.ocr_render_dpi / 72
            matrix = fitz.Matrix(scale, scale)
            images: list[Image.Image] = []
            for page in pdf:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes("png")
                images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            return images
    except Exception as exc:
        raise OcrError("PDF rendering for OCR failed") from exc


def _extract_words_with_confidence(image: Image.Image) -> tuple[str, list[float]]:
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words: list[str] = []
    confidences: list[float] = []

    for raw_text, raw_confidence in zip(data.get("text", []), data.get("conf", []), strict=False):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue
        words.append(text)
        confidences.append(confidence / 100)

    return " ".join(words).strip(), confidences


def extract_scanned_pdf_with_tesseract(content: bytes) -> OcrResult:
    configure_tesseract()
    page_images = _page_images_from_pdf(content)
    if not page_images:
        raise OcrError("OCR document did not contain any pages")

    page_texts: list[str] = []
    all_confidences: list[float] = []
    for image in page_images:
        page_text, page_confidences = _extract_words_with_confidence(image)
        if page_text:
            page_texts.append(page_text)
        all_confidences.extend(page_confidences)

    text = "\n\n".join(page_texts).strip()
    if not text:
        raise OcrError("Tesseract OCR did not extract text")

    confidence = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0
    review_required = confidence < settings.ocr_confidence_review_threshold
    return OcrResult(
        text=text,
        engine="tesseract",
        page_count=len(page_images),
        non_empty_page_count=len(page_texts),
        confidence=confidence,
        review_required=review_required,
    )


def _storage_path_for(document: Document) -> Path:
    raw_storage_path = document.document_metadata.get("storage_path")
    if not raw_storage_path:
        raise OcrError("Document metadata is missing storage_path")
    return Path(raw_storage_path)


def _ensure_ocr_supported(document: Document) -> None:
    if not document.ocr_required and document.document_type not in {
        DocumentType.SCANNED_PDF,
        DocumentType.HANDWRITTEN_NOTE,
    }:
        raise OcrError("Document is not marked for OCR")
    if document.mime_type != "application/pdf":
        raise OcrError("OCR fallback currently supports scanned PDF uploads only")


def process_ocr_ingestion_job(db: Session, job_id: uuid.UUID) -> OcrResult:
    job = db.scalar(
        select(IngestionJob)
        .options(selectinload(IngestionJob.document))
        .where(IngestionJob.id == job_id)
    )
    if job is None:
        raise OcrError("Ingestion job was not found")

    document = job.document
    started_at = datetime.now(UTC)
    job.status = IngestionJobStatus.RUNNING
    job.stage = "ocr_extracting"
    job.attempts += 1
    job.started_at = started_at
    job.error_message = None
    document.status = DocumentStatus.EXTRACTING
    db.commit()

    try:
        _ensure_ocr_supported(document)
        content = read_encrypted_document(
            storage_path=_storage_path_for(document),
            document_id=document.id,
        )
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != document.checksum_sha256:
            raise OcrError("Stored document checksum does not match metadata")

        result = extract_scanned_pdf_with_tesseract(content)
        processed_path = write_processed_text(document.id, result.text)
        extracted_at = datetime.now(UTC)
        ocr_metadata = {
            "engine": result.engine,
            "extracted_at": extracted_at.isoformat(),
            "confidence": result.confidence,
            "review_threshold": settings.ocr_confidence_review_threshold,
            "review_required": result.review_required,
            "text_path": str(processed_path),
            "text_uri": f"local-processed://{processed_path.name}",
            "char_count": result.char_count,
            "page_count": result.page_count,
            "non_empty_page_count": result.non_empty_page_count,
            "checksum_verified": True,
        }

        document.ocr_engine = result.engine
        document.ocr_confidence = result.confidence
        document.document_metadata = {
            **document.document_metadata,
            "ocr": ocr_metadata,
            "extraction": {
                "extractor": result.engine,
                **ocr_metadata,
            },
        }
        document.status = (
            DocumentStatus.REVIEW_REQUIRED if result.review_required else DocumentStatus.PROCESSED
        )
        job.status = IngestionJobStatus.SUCCEEDED
        job.stage = "ocr_review_required" if result.review_required else "ocr_extracted"
        job.finished_at = extracted_at
        job.job_metadata = {
            **job.job_metadata,
            "ocr": ocr_metadata,
        }
        db.commit()
        return result
    except (OcrError, StorageReadError, ExtractionError) as exc:
        failed_at = datetime.now(UTC)
        document.status = DocumentStatus.FAILED
        job.status = IngestionJobStatus.FAILED
        job.stage = "ocr_failed"
        job.error_message = str(exc)
        job.finished_at = failed_at
        job.job_metadata = {
            **job.job_metadata,
            "failed_at": failed_at.isoformat(),
        }
        db.commit()
        raise OcrError(str(exc)) from exc


def process_queued_ocr_jobs(db: Session, *, limit: int = 25) -> list[uuid.UUID]:
    jobs = db.scalars(
        select(IngestionJob)
        .join(IngestionJob.document)
        .where(IngestionJob.status == IngestionJobStatus.QUEUED)
        .where(Document.ocr_required.is_(True))
        .order_by(IngestionJob.created_at)
        .limit(limit)
    ).all()

    processed_job_ids: list[uuid.UUID] = []
    for job in jobs:
        process_ocr_ingestion_job(db, job.id)
        processed_job_ids.append(job.id)
    return processed_job_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OCR text for queued scanned documents.")
    parser.add_argument("--job-id", type=uuid.UUID)
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        if args.job_id:
            process_ocr_ingestion_job(db, args.job_id)
            print(f"Processed OCR ingestion job {args.job_id}")
            return

        processed_job_ids = process_queued_ocr_jobs(db, limit=args.limit)
        print(f"Processed {len(processed_job_ids)} OCR ingestion jobs")


if __name__ == "__main__":
    main()
