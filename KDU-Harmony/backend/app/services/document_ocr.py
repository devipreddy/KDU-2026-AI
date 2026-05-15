from __future__ import annotations

import argparse
import base64
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus, DocumentType, IngestionJobStatus
from app.models.ingestion_job import IngestionJob
from app.services.clinical_metadata import (
    apply_clinical_metadata_to_document,
    extract_clinical_metadata,
)
from app.services.document_chunking import chunk_document_text, delete_existing_chunks
from app.services.document_extraction import ExtractionError, write_processed_text
from app.services.document_storage import StorageReadError, read_encrypted_document
from app.services.embedding_pipeline import (
    EmbeddingEncoder,
    EmbeddingPipelineError,
    index_document_chunks,
)
from app.services.phi_tokenization import tokenize_phi_for_document
from app.services.text_normalization import normalize_medical_text

QIANFAN_OCR_ENGINE = "openrouter:qianfan-ocr-fast"
QIANFAN_OCR_SYSTEM_PROMPT = """You are a medical document OCR and document-intelligence engine.

Task:
Extract all readable text from the supplied document image with maximum fidelity.

Rules:
1. Preserve the document's reading order, headings, labels, tables, line breaks, punctuation,
   capitalization, abbreviations, medication names, dosages, dates, identifiers, stamps, and
   signatures as closely as possible.
2. Do not silently correct clear source text. If a word is visibly misspelled in the document,
   keep the visible spelling. If a word is blurry or partially obscured, infer the most likely
   spelling from nearby context and mark it as `[unclear: inferred text]`.
3. Do not redact PHI or identifiers. The downstream healthcare pipeline performs PHI tokenization
   after OCR. Your job is accurate extraction.
4. Identify patients, doctors or prescribers, medical facilities or hospitals, dates, medications,
   prescription instructions, IDs or MRNs, and other clinically relevant fields when present.
5. If a field is absent or unreadable, write `Not visible` rather than inventing content.
6. Return only Markdown in the exact structure below.

Output format:
## Transcription
Preserve the original structure here. Use Markdown tables only when the document has a real table
or form-like paired fields.

## Identified entities
- Patient(s):
- Doctor(s) / Prescriber(s):
- Hospital(s) / Facility:
- Dates:
- Medications / Orders:
- Identifiers:

## OCR uncertainty notes
- List blurry, inferred, ambiguous, or low-confidence items. If none, write `None noted.`

## Summary
Provide a clear explanatory summary of what the document appears to be, including the patient,
facility, clinician, clinical purpose, medications or orders, and any important caveats."""


class OcrError(ValueError):
    """Raised when cloud OCR cannot extract text from an OCR-required document."""


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_data_url: str


@dataclass(frozen=True)
class OcrResult:
    text: str
    engine: str
    page_count: int
    non_empty_page_count: int
    confidence: float
    review_required: bool
    engine_attempts: list[dict[str, Any]]

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class PageOcrText:
    text: str
    model: str


def render_pdf_pages_for_ocr(content: bytes) -> list[RenderedPage]:
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            scale = settings.ocr_render_dpi / 72
            matrix = fitz.Matrix(scale, scale)
            pages: list[RenderedPage] = []
            for page_index, page in enumerate(pdf, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes("png")
                encoded = base64.b64encode(image_bytes).decode("ascii")
                pages.append(
                    RenderedPage(
                        page_number=page_index,
                        image_data_url=f"data:image/png;base64,{encoded}",
                    )
                )
            return pages
    except Exception as exc:
        raise OcrError(f"PDF rendering for OpenRouter OCR failed: {exc}") from exc


def extract_scanned_pdf(content: bytes) -> OcrResult:
    return extract_scanned_pdf_with_qianfan(content)


def extract_scanned_pdf_with_qianfan(content: bytes) -> OcrResult:
    pages = render_pdf_pages_for_ocr(content)
    if not pages:
        raise OcrError("OCR document did not contain any pages")

    attempts: list[dict[str, Any]] = []
    max_workers = ocr_worker_count(page_count=len(pages))
    try:
        with httpx.Client(timeout=settings.openrouter_ocr_timeout_seconds) as client:
            if max_workers == 1:
                page_results = [call_openrouter_qianfan_ocr(page, client=client) for page in pages]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    page_results = list(
                        executor.map(
                            lambda page: call_openrouter_qianfan_ocr(page, client=client),
                            pages,
                        )
                    )
    except OcrError:
        raise

    page_outputs = [
        f"<!-- page:{page.page_number} -->\n{page_result.text}"
        for page, page_result in zip(pages, page_results, strict=True)
        if page_result.text
    ]
    successful_models = sorted({page_result.model for page_result in page_results})

    text = "\n\n".join(page_outputs).strip()
    if not text:
        raise OcrError("OpenRouter Qianfan OCR returned an empty transcription")

    attempts.append(
        {
            "engine": QIANFAN_OCR_ENGINE,
            "model": successful_models[0] if len(successful_models) == 1 else "mixed",
            "models": successful_models,
            "status": "succeeded",
            "page_count": len(pages),
            "max_workers": max_workers,
        }
    )
    return OcrResult(
        text=text,
        engine=QIANFAN_OCR_ENGINE,
        page_count=len(pages),
        non_empty_page_count=len(page_outputs),
        confidence=1.0,
        review_required=False,
        engine_attempts=attempts,
    )


def ocr_worker_count(*, page_count: int) -> int:
    return max(1, min(page_count, settings.openrouter_ocr_max_workers))


def call_openrouter_qianfan_ocr(
    page: RenderedPage,
    *,
    client: httpx.Client | None = None,
) -> PageOcrText:
    if not settings.openrouter_api_key:
        raise OcrError(
            "OPENROUTER_API_KEY is not configured. Add it to backend/.env and restart the API."
        )

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": settings.project_name,
    }
    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    model_errors: list[str] = []

    for model in openrouter_ocr_model_candidates():
        payload = build_openrouter_ocr_payload(page, model=model)
        try:
            response = post_openrouter_ocr_request(
                url=url,
                headers=headers,
                payload=payload,
                client=client,
            )
        except OcrError as exc:
            model_errors.append(f"{model}: {exc}")
            if should_try_next_ocr_model(str(exc)):
                continue
            raise

        if response.status_code >= 400:
            error_message = (
                "OpenRouter Qianfan OCR returned "
                f"HTTP {response.status_code} for model {model}: {safe_response_body(response)}"
            )
            model_errors.append(f"{model}: {error_message}")
            if should_try_next_ocr_model(error_message):
                continue
            raise OcrError(error_message)

        try:
            body = response.json()
        except ValueError as exc:
            raise OcrError(
                "OpenRouter Qianfan OCR returned non-JSON response "
                f"for model {model}: {safe_response_body(response)}"
            ) from exc

        return PageOcrText(
            text=extract_openrouter_message_text(body),
            model=model,
        )

    raise OcrError(
        "OpenRouter Qianfan OCR failed for all configured models: "
        + " | ".join(model_errors)
    )


def post_openrouter_ocr_request(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    client: httpx.Client | None,
) -> httpx.Response:
    try:
        if client is None:
            with httpx.Client(timeout=settings.openrouter_ocr_timeout_seconds) as owned_client:
                return owned_client.post(url, headers=headers, json=payload)
        return client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise OcrError(
            f"OpenRouter Qianfan OCR timed out after "
            f"{settings.openrouter_ocr_timeout_seconds:g} seconds"
        ) from exc
    except httpx.HTTPError as exc:
        raise OcrError(f"OpenRouter Qianfan OCR request failed: {exc}") from exc


def openrouter_ocr_model_candidates() -> list[str]:
    return ordered_model_names(
        [
            settings.openrouter_ocr_model,
            *settings.openrouter_ocr_fallback_models.split(","),
        ]
    )


def ordered_model_names(models: list[str]) -> list[str]:
    ordered: list[str] = []
    for model in models:
        cleaned = model.strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered


def should_try_next_ocr_model(error_message: str) -> bool:
    normalized = error_message.lower()
    return (
        "http 404" in normalized
        or "no endpoints found" in normalized
        or "model not found" in normalized
    )


def build_openrouter_ocr_payload(
    page: RenderedPage,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    return {
        "model": model or settings.openrouter_ocr_model,
        "messages": [
            {
                "role": "system",
                "content": QIANFAN_OCR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract this medical document page. Follow the system format exactly. "
                            f"This is page {page.page_number}."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": page.image_data_url,
                        },
                    },
                ],
            },
        ],
        "max_tokens": settings.openrouter_ocr_max_tokens,
        "temperature": 0,
    }


def extract_openrouter_message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OcrError(f"OpenRouter Qianfan OCR response had no choices: {body}")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise OcrError(f"OpenRouter Qianfan OCR response had no message: {body}")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts).strip()
    raise OcrError(f"OpenRouter Qianfan OCR response content was not text: {body}")


def safe_response_body(response: httpx.Response, *, max_chars: int = 1200) -> str:
    text = response.text.strip()
    return text[:max_chars] if text else "<empty body>"


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
        raise OcrError("OpenRouter Qianfan OCR currently supports scanned PDF uploads only")


def process_ocr_ingestion_job(
    db: Session,
    job_id: uuid.UUID,
    *,
    index_after_extraction: bool | None = None,
    chunk_after_extraction: bool = True,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
) -> OcrResult:
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
    job.stage = "openrouter_ocr_extracting"
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

        result = extract_scanned_pdf(content)
        normalization = normalize_medical_text(
            result.text,
            apply_corrections=False,
            preserve_structure=True,
        )
        phi_tokenization = tokenize_phi_for_document(
            db,
            document=document,
            text=normalization.text,
        )
        clinical_metadata = extract_clinical_metadata(phi_tokenization.text)
        apply_clinical_metadata_to_document(document, clinical_metadata)
        chunking_metadata: dict[str, Any] = {
            "status": "pending_review",
            "parent_chunk_count": 0,
            "child_chunk_count": 0,
            "total_chunk_count": 0,
        }
        if chunk_after_extraction:
            chunking = chunk_document_text(
                db,
                document=document,
                text=phi_tokenization.text,
                clinical_metadata=clinical_metadata,
                ocr_confidence=result.confidence,
            )
            chunking_metadata = chunking.to_metadata()
        else:
            delete_existing_chunks(db, document)
        normalized_result = OcrResult(
            text=phi_tokenization.text,
            engine=result.engine,
            page_count=result.page_count,
            non_empty_page_count=result.non_empty_page_count,
            confidence=result.confidence,
            review_required=result.review_required,
            engine_attempts=result.engine_attempts,
        )
        processed_path = write_processed_text(document.id, normalized_result.text)
        extracted_at = datetime.now(UTC)
        review_gate = {
            "required": not chunk_after_extraction or normalized_result.review_required,
            "status": "pending"
            if not chunk_after_extraction or normalized_result.review_required
            else "not_required",
            "requested_at": extracted_at.isoformat(),
        }
        ocr_metadata = {
            "engine": normalized_result.engine,
            "model": ocr_result_model(normalized_result),
            "configured_model": settings.openrouter_ocr_model,
            "model_candidates": openrouter_ocr_model_candidates(),
            "provider": "openrouter",
            "system_prompt": QIANFAN_OCR_SYSTEM_PROMPT,
            "engine_attempts": normalized_result.engine_attempts,
            "extracted_at": extracted_at.isoformat(),
            "confidence": normalized_result.confidence,
            "confidence_source": "cloud_model_no_numeric_confidence",
            "review_required": normalized_result.review_required,
            "text_path": str(processed_path),
            "text_uri": f"local-processed://{processed_path.name}",
            "char_count": normalized_result.char_count,
            "raw_char_count": result.char_count,
            "normalized_char_count": len(normalization.text),
            "page_count": normalized_result.page_count,
            "non_empty_page_count": normalized_result.non_empty_page_count,
            "checksum_verified": True,
            "normalization": asdict(normalization.stats),
            "phi": phi_tokenization.metadata_summary(),
            "clinical": clinical_metadata.to_metadata(),
            "chunking": chunking_metadata,
            "review_gate": review_gate,
        }

        document.ocr_engine = normalized_result.engine
        document.ocr_confidence = normalized_result.confidence
        document.document_metadata = {
            **document.document_metadata,
            "review_gate": review_gate,
            "ocr": ocr_metadata,
            "extraction": {
                "extractor": result.engine,
                **ocr_metadata,
            },
        }
        document.status = (
            DocumentStatus.PROCESSED
            if chunk_after_extraction and not result.review_required
            else DocumentStatus.REVIEW_REQUIRED
        )
        job.status = IngestionJobStatus.SUCCEEDED
        job.stage = (
            "openrouter_ocr_extracted"
            if chunk_after_extraction and not result.review_required
            else "openrouter_ocr_review_required"
        )
        job.finished_at = extracted_at
        job.job_metadata = {
            **job.job_metadata,
            "ocr": ocr_metadata,
        }
        db.commit()
        if (
            chunk_after_extraction
            and should_index_after_ocr(index_after_extraction)
            and not normalized_result.review_required
        ):
            indexing_result = index_document_chunks(
                db,
                document_id=document.id,
                ingestion_job=job,
                encoder=encoder,
                collection=collection,
            )
            document.document_metadata = {
                **(document.document_metadata or {}),
                "ocr": {
                    **((document.document_metadata or {}).get("ocr") or {}),
                    "indexing": indexing_result.to_metadata(),
                },
                "extraction": {
                    **((document.document_metadata or {}).get("extraction") or {}),
                    "indexing": indexing_result.to_metadata(),
                },
            }
            db.commit()
        return normalized_result
    except EmbeddingPipelineError as exc:
        raise OcrError(f"Chunk indexing failed: {exc}") from exc
    except (OcrError, StorageReadError, ExtractionError) as exc:
        failed_at = datetime.now(UTC)
        document.status = DocumentStatus.FAILED
        job.status = IngestionJobStatus.FAILED
        job.stage = "openrouter_ocr_failed"
        job.error_message = str(exc)
        job.finished_at = failed_at
        job.job_metadata = {
            **job.job_metadata,
            "failed_at": failed_at.isoformat(),
            "ocr_error": str(exc),
            "ocr_engine": QIANFAN_OCR_ENGINE,
            "ocr_model": settings.openrouter_ocr_model,
            "ocr_model_candidates": openrouter_ocr_model_candidates(),
        }
        db.commit()
        raise OcrError(str(exc)) from exc
    except Exception as exc:
        failed_at = datetime.now(UTC)
        error_message = f"Unexpected OpenRouter Qianfan OCR failure: {exc}"
        document.status = DocumentStatus.FAILED
        job.status = IngestionJobStatus.FAILED
        job.stage = "openrouter_ocr_failed"
        job.error_message = error_message
        job.finished_at = failed_at
        job.job_metadata = {
            **job.job_metadata,
            "failed_at": failed_at.isoformat(),
            "ocr_error": error_message,
            "ocr_engine": QIANFAN_OCR_ENGINE,
            "ocr_model": settings.openrouter_ocr_model,
            "ocr_model_candidates": openrouter_ocr_model_candidates(),
        }
        db.commit()
        raise OcrError(error_message) from exc


def should_index_after_ocr(index_after_extraction: bool | None) -> bool:
    return settings.index_on_ingestion if index_after_extraction is None else index_after_extraction


def ocr_result_model(result: OcrResult) -> str:
    for attempt in reversed(result.engine_attempts):
        model = attempt.get("model")
        if isinstance(model, str) and model:
            return model
    return settings.openrouter_ocr_model


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
    parser = argparse.ArgumentParser(
        description="Extract OCR text for queued scanned documents with OpenRouter Qianfan OCR."
    )
    parser.add_argument("--job-id", type=uuid.UUID)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--show-system-prompt",
        action="store_true",
        help="Print the Qianfan OCR system prompt and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_system_prompt:
        print(QIANFAN_OCR_SYSTEM_PROMPT)
        return

    with SessionLocal() as db:
        if args.job_id:
            process_ocr_ingestion_job(db, args.job_id)
            print(f"Processed OCR ingestion job {args.job_id}")
            return

        processed_job_ids = process_queued_ocr_jobs(db, limit=args.limit)
        print(f"Processed {len(processed_job_ids)} OCR ingestion jobs")


if __name__ == "__main__":
    main()
