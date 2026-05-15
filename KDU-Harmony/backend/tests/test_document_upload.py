from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.seed import DEMO_PASSWORD, seed_auth_data
from app.db.session import get_db
from app.main import create_app
from app.models.document import Document
from app.models.ingestion_job import IngestionJob


@dataclass
class UploadApp:
    client: TestClient
    session_local: sessionmaker[Session]
    storage_root: Path


@pytest.fixture()
def upload_app(tmp_path: Path) -> Generator[UploadApp, None, None]:
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
    settings.document_storage_root = tmp_path

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield UploadApp(
            client=test_client,
            session_local=testing_session_local,
            storage_root=tmp_path,
        )

    app.dependency_overrides.clear()
    settings.document_storage_root = original_storage_root


def auth_header(upload_app: UploadApp, email: str = "records@example.com") -> dict[str, str]:
    response = upload_app.client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_records_staff_can_upload_text_document(upload_app: UploadApp) -> None:
    content = b"Patient has controlled hypertension and needs follow-up."

    response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(upload_app),
        data={
            "patient_ref": "PATIENT_REF_0001",
            "visit_id": "VISIT-LOCAL-0001",
            "document_type": "clinical_note",
            "hospital": "Harmony General Hospital",
            "physician": "Dr. Asha Raman",
            "department": "Cardiology",
            "diagnosis": "Hypertension",
            "icd_codes": '["I10"]',
            "sensitivity_level": "high",
            "metadata": '{"source_system":"unit-test"}',
        },
        files={"file": ("note.txt", content, "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["patient_ref"] == "PATIENT_REF_0001"
    assert body["document_type"] == "clinical_note"
    assert body["status"] == "uploaded"
    assert body["mime_type"] == "text/plain"
    assert body["is_encrypted"] is True
    assert body["ocr_required"] is False

    encrypted_files = list(upload_app.storage_root.rglob("*.enc"))
    assert len(encrypted_files) == 1
    encrypted_blob = encrypted_files[0].read_bytes()
    assert content not in encrypted_blob

    with upload_app.session_local() as db:
        document_id = UUID(body["id"])
        document = db.scalar(select(Document).where(Document.id == document_id))
        jobs = db.scalars(select(IngestionJob).where(IngestionJob.document_id == document_id)).all()

    assert document is not None
    assert document.checksum_sha256 == body["checksum_sha256"]
    assert document.document_metadata["source_system"] == "unit-test"
    assert document.document_metadata["size_bytes"] == len(content)
    assert len(jobs) == 1
    assert jobs[0].status.value == "queued"
    assert jobs[0].stage == "uploaded"


def test_duplicate_upload_reuses_existing_document(upload_app: UploadApp) -> None:
    headers = auth_header(upload_app)
    content = b"Patient has controlled hypertension and needs follow-up."

    first_response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=headers,
        data={"patient_ref": "PATIENT_REF_DUPLICATE"},
        files={"file": ("duplicate-note.txt", content, "text/plain")},
    )
    second_response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=headers,
        data={"patient_ref": "PATIENT_REF_DUPLICATE"},
        files={"file": ("duplicate-note.txt", content, "text/plain")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    first_body = first_response.json()
    second_body = second_response.json()
    assert second_body["id"] == first_body["id"]
    assert second_body["ingestion_job_id"] == first_body["ingestion_job_id"]

    with upload_app.session_local() as db:
        document_count = db.scalar(select(func.count(Document.id)))
        job_count = db.scalar(select(func.count(IngestionJob.id)))

    assert document_count == 1
    assert job_count == 1
    assert len(list(upload_app.storage_root.rglob("*.enc"))) == 1


def test_pdf_upload_defaults_to_typed_pdf(upload_app: UploadApp) -> None:
    content = b"%PDF-1.7\nsynthetic pdf bytes\n%%EOF"

    response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(upload_app, "doctor@example.com"),
        data={"patient_ref": "PATIENT_REF_0002"},
        files={"file": ("scan.pdf", content, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_type"] == "typed_pdf"
    assert body["mime_type"] == "application/pdf"


def test_document_list_can_filter_ingestion_dashboard_uploads(upload_app: UploadApp) -> None:
    headers = auth_header(upload_app)
    first_response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=headers,
        data={
            "patient_ref": "PATIENT_REF_0101",
            "metadata": '{"upload_source":"ingestion_dashboard","dataset_path":"Dataset/a.pdf"}',
        },
        files={"file": ("dashboard.pdf", b"%PDF-1.7\ndashboard\n%%EOF", "application/pdf")},
    )
    second_response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=headers,
        data={
            "patient_ref": "PATIENT_REF_0102",
            "metadata": '{"source":"synthetic_dataset"}',
        },
        files={"file": ("synthetic.pdf", b"%PDF-1.7\nsynthetic\n%%EOF", "application/pdf")},
    )
    assert first_response.status_code == 201
    assert second_response.status_code == 201

    response = upload_app.client.get(
        "/api/v1/documents?source=ingestion_dashboard&limit=500",
        headers=headers,
    )

    assert response.status_code == 200
    documents = response.json()
    assert [document["file_name"] for document in documents] == ["dashboard.pdf"]


def test_invalid_pdf_signature_is_rejected(upload_app: UploadApp) -> None:
    response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(upload_app),
        data={"patient_ref": "PATIENT_REF_0003"},
        files={"file": ("bad.pdf", b"not actually a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_nurse_cannot_upload_document(upload_app: UploadApp) -> None:
    response = upload_app.client.post(
        "/api/v1/documents/upload",
        headers=auth_header(upload_app, "nurse@example.com"),
        data={"patient_ref": "PATIENT_REF_0004"},
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 403
