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
from app.models.ingestion_job import IngestionJob
from app.models.phi_mapping import PhiMapping
from app.services.document_extraction import (
    ExtractionError,
    process_ingestion_job,
    process_queued_ingestion_jobs,
)


@dataclass
class ExtractionApp:
    client: TestClient
    session_local: sessionmaker[Session]
    storage_root: Path
    processed_root: Path


@pytest.fixture()
def extraction_app(tmp_path: Path) -> Generator[ExtractionApp, None, None]:
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
    settings.document_storage_root = tmp_path / "storage"
    settings.processed_text_root = tmp_path / "processed"

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield ExtractionApp(
            client=test_client,
            session_local=testing_session_local,
            storage_root=settings.document_storage_root,
            processed_root=settings.processed_text_root,
        )

    app.dependency_overrides.clear()
    settings.document_storage_root = original_storage_root
    settings.processed_text_root = original_processed_root


def auth_header(extraction_app: ExtractionApp) -> dict[str, str]:
    response = extraction_app.client.post(
        "/api/v1/auth/login",
        json={"email": "records@example.com", "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def upload_file(
    extraction_app: ExtractionApp,
    *,
    filename: str,
    content: bytes,
    content_type: str,
    document_type: str | None = None,
) -> dict:
    data = {"patient_ref": "PATIENT_REF_0100"}
    if document_type:
        data["document_type"] = document_type

    response = extraction_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(extraction_app),
        data=data,
        files={"file": (filename, content, content_type)},
    )
    assert response.status_code == 201
    return response.json()


def build_pdf_bytes(text: str) -> bytes:
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
        return pdf.tobytes()


def test_text_upload_can_be_extracted_to_processed_text(extraction_app: ExtractionApp) -> None:
    upload = upload_file(
        extraction_app,
        filename="clinical-note.txt",
        content=(
            b"Patient Name: John Smith\n"
            b"DOB: 01/23/1980\n"
            b"MRN: MRN-100100\n"
            b"Phone: (555) 123-4567\n"
            b"Address: 12 Oak Street, Boston, MA 02118\n"
            b"assessment - Type 2 diabetes is stable.\n"
            b"Plan:: Continue Metforrnin.\n"
            b"Diagnosis: Hvpcrtension"
        ),
        content_type="text/plain",
    )

    with extraction_app.session_local() as db:
        result = process_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert result.extractor == "plain_text"
    assert "Continue Metformin" in result.text
    assert "Diagnosis: Hypertension" in result.text
    assert "John Smith" not in result.text
    assert "MRN-100100" not in result.text
    assert "[PATIENT_REF_0100]" in result.text
    assert "[DOB_0100_" in result.text
    assert "[MRN_0100_" in result.text

    processed_files = list(extraction_app.processed_root.glob("*.txt"))
    assert len(processed_files) == 1
    processed_text = processed_files[0].read_text(encoding="utf-8")
    assert "Type 2 diabetes" in processed_text
    assert "Metforrnin" not in processed_text
    assert "John Smith" not in processed_text
    assert "12 Oak Street" not in processed_text

    with extraction_app.session_local() as db:
        document = db.scalar(select(Document).where(Document.id == UUID(upload["id"])))
        job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(upload["ingestion_job_id"]))
        )
        phi_mappings = db.scalars(
            select(PhiMapping).where(PhiMapping.patient_ref == "PATIENT_REF_0100")
        ).all()

    assert document is not None
    assert document.status.value == "processed"
    assert document.document_metadata["extraction"]["extractor"] == "plain_text"
    assert document.document_metadata["extraction"]["checksum_verified"] is True
    assert (
        document.document_metadata["extraction"]["normalization"]["medical_corrections"][
            "Hypertension"
        ]
        == 1
    )
    assert document.document_metadata["extraction"]["phi"]["entity_counts"]["PATIENT_NAME"] == 1
    assert document.document_metadata["extraction"]["phi"]["entity_counts"]["MRN"] == 1
    assert len(phi_mappings) == 5
    assert all("John Smith" not in mapping.encrypted_value for mapping in phi_mappings)
    assert job is not None
    assert job.status.value == "succeeded"
    assert job.stage == "text_extracted"


def test_typed_pdf_upload_can_be_extracted_with_pymupdf(extraction_app: ExtractionApp) -> None:
    pdf_text = "Cardiology note: atrial fibrillation treated with metoprolol."
    upload = upload_file(
        extraction_app,
        filename="typed-record.pdf",
        content=build_pdf_bytes(pdf_text),
        content_type="application/pdf",
        document_type="typed_pdf",
    )

    with extraction_app.session_local() as db:
        result = process_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert result.extractor == "pymupdf"
    assert result.page_count == 1
    assert result.non_empty_page_count == 1
    assert "atrial fibrillation" in result.text


def test_queued_extraction_skips_ocr_required_documents(extraction_app: ExtractionApp) -> None:
    typed_upload = upload_file(
        extraction_app,
        filename="typed-record.txt",
        content=b"Typed text should be extracted.",
        content_type="text/plain",
    )
    scanned_upload = upload_file(
        extraction_app,
        filename="scanned-record.pdf",
        content=build_pdf_bytes("This will be routed to OCR later."),
        content_type="application/pdf",
        document_type="scanned_pdf",
    )

    with extraction_app.session_local() as db:
        processed_job_ids = process_queued_ingestion_jobs(db, limit=10)
        scanned_job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(scanned_upload["ingestion_job_id"]))
        )

    assert processed_job_ids == [UUID(typed_upload["ingestion_job_id"])]
    assert scanned_job is not None
    assert scanned_job.status.value == "queued"


def test_direct_extraction_fails_for_ocr_required_document(extraction_app: ExtractionApp) -> None:
    upload = upload_file(
        extraction_app,
        filename="handwritten.pdf",
        content=build_pdf_bytes("OCR later."),
        content_type="application/pdf",
        document_type="handwritten_note",
    )

    with extraction_app.session_local() as db:
        with pytest.raises(ExtractionError, match="requires OCR"):
            process_ingestion_job(db, UUID(upload["ingestion_job_id"]))

        job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(upload["ingestion_job_id"]))
        )

    assert job is not None
    assert job.status.value == "failed"
    assert job.stage == "text_extraction_failed"
