from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.seed import DEMO_PASSWORD, seed_auth_data
from app.db.session import get_db
from app.main import create_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.ingestion_job import IngestionJob
from app.services.document_ocr import (
    OcrError,
    process_ocr_ingestion_job,
    process_queued_ocr_jobs,
)


@dataclass
class OcrApp:
    client: TestClient
    session_local: sessionmaker[Session]
    processed_root: Path


@pytest.fixture()
def ocr_app(tmp_path: Path) -> Generator[OcrApp, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with testing_session_local() as db:
        seed_auth_data(db)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session_local() as db:
            yield db

    original_storage_root = settings.document_storage_root
    original_processed_root = settings.processed_text_root
    original_threshold = settings.ocr_confidence_review_threshold
    settings.document_storage_root = tmp_path / "storage"
    settings.processed_text_root = tmp_path / "processed"
    settings.ocr_confidence_review_threshold = 0.80

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield OcrApp(
            client=test_client,
            session_local=testing_session_local,
            processed_root=settings.processed_text_root,
        )

    app.dependency_overrides.clear()
    settings.document_storage_root = original_storage_root
    settings.processed_text_root = original_processed_root
    settings.ocr_confidence_review_threshold = original_threshold


def auth_header(ocr_app: OcrApp) -> dict[str, str]:
    response = ocr_app.client.post(
        "/api/v1/auth/login",
        json={"email": "records@example.com", "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def build_pdf_bytes(text: str) -> bytes:
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        return pdf.tobytes()


def upload_scanned_pdf(ocr_app: OcrApp) -> dict:
    response = ocr_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(ocr_app),
        data={
            "patient_ref": "PATIENT_REF_0200",
            "document_type": "scanned_pdf",
        },
        files={
            "file": (
                "scanned.pdf",
                build_pdf_bytes("Image-like scanned source"),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["ocr_required"] is True
    return body


def mock_tesseract(monkeypatch: pytest.MonkeyPatch, *, confidence: int) -> None:
    def fake_image_to_data(*args, **kwargs):
        return {
            "text": ["", "Diagnosis", "Hvpcrtension", "Bcta", "blockers"],
            "conf": ["-1", str(confidence), str(confidence), str(confidence), str(confidence)],
        }

    monkeypatch.setattr("app.services.document_ocr.pytesseract.image_to_data", fake_image_to_data)


def test_scanned_pdf_ocr_success_updates_document_and_job(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_tesseract(monkeypatch, confidence=91)
    upload = upload_scanned_pdf(ocr_app)

    with ocr_app.session_local() as db:
        result = process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert result.engine == "tesseract"
    assert result.confidence == 0.91
    assert result.review_required is False
    assert "Diagnosis Hypertension Beta blockers" in result.text

    processed_files = list(ocr_app.processed_root.glob("*.txt"))
    assert len(processed_files) == 1
    assert "Hypertension Beta blockers" in processed_files[0].read_text(encoding="utf-8")

    with ocr_app.session_local() as db:
        document = db.scalar(select(Document).where(Document.id == UUID(upload["id"])))
        job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(upload["ingestion_job_id"]))
        )
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == UUID(upload["id"]))
            .order_by(DocumentChunk.chunk_index)
        ).all()

    assert document is not None
    assert document.status.value == "processed"
    assert float(document.ocr_confidence) == 0.91
    assert document.ocr_engine == "tesseract"
    assert document.document_metadata["ocr"]["review_required"] is False
    assert (
        document.document_metadata["ocr"]["normalization"]["medical_corrections"]["Hypertension"]
        == 1
    )
    clinical = document.document_metadata["ocr"]["clinical"]
    assert clinical["diagnoses"] == ["Hypertension"]
    assert clinical["medications"] == ["Beta blockers"]
    chunking = document.document_metadata["ocr"]["chunking"]
    assert chunking["parent_chunk_count"] == 1
    assert chunking["child_chunk_count"] == 1
    assert len(chunks) == 2
    assert chunks[0].section == "Document"
    assert chunks[1].parent_chunk_id == chunks[0].id
    assert float(chunks[1].ocr_confidence) == 0.91
    assert job is not None
    assert job.status.value == "succeeded"
    assert job.stage == "ocr_extracted"


def test_low_confidence_ocr_routes_document_to_review(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_tesseract(monkeypatch, confidence=62)
    upload = upload_scanned_pdf(ocr_app)

    with ocr_app.session_local() as db:
        result = process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert result.confidence == 0.62
    assert result.review_required is True

    with ocr_app.session_local() as db:
        document = db.scalar(select(Document).where(Document.id == UUID(upload["id"])))
        job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(upload["ingestion_job_id"]))
        )

    assert document is not None
    assert document.status.value == "review_required"
    assert float(document.ocr_confidence) == 0.62
    assert document.document_metadata["ocr"]["review_required"] is True
    assert job is not None
    assert job.status.value == "succeeded"
    assert job.stage == "ocr_review_required"


def test_queued_ocr_processor_processes_only_ocr_required_jobs(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_tesseract(monkeypatch, confidence=88)
    scanned_upload = upload_scanned_pdf(ocr_app)
    text_response = ocr_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(ocr_app),
        data={"patient_ref": "PATIENT_REF_0201"},
        files={"file": ("note.txt", b"Plain text stays in typed queue.", "text/plain")},
    )
    assert text_response.status_code == 201
    text_upload = text_response.json()

    with ocr_app.session_local() as db:
        processed_job_ids = process_queued_ocr_jobs(db, limit=10)
        text_job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(text_upload["ingestion_job_id"]))
        )

    assert processed_job_ids == [UUID(scanned_upload["ingestion_job_id"])]
    assert text_job is not None
    assert text_job.status.value == "queued"


def test_ocr_processor_rejects_non_ocr_document(ocr_app: OcrApp) -> None:
    text_response = ocr_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(ocr_app),
        data={"patient_ref": "PATIENT_REF_0202"},
        files={"file": ("note.txt", b"Not OCR.", "text/plain")},
    )
    assert text_response.status_code == 201

    with ocr_app.session_local() as db:
        with pytest.raises(OcrError, match="not marked for OCR"):
            process_ocr_ingestion_job(db, UUID(text_response.json()["ingestion_job_id"]))
