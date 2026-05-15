from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.seed import DEMO_PASSWORD, seed_auth_data
from app.db.session import get_db
from app.main import create_app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import ChunkIndexingStatus, RoleName
from app.models.ingestion_job import IngestionJob
from app.models.user import User
from app.services.document_ocr import (
    QIANFAN_OCR_ENGINE,
    QIANFAN_OCR_SYSTEM_PROMPT,
    OcrError,
    RenderedPage,
    build_openrouter_ocr_payload,
    extract_scanned_pdf,
    process_ocr_ingestion_job,
    process_queued_ocr_jobs,
)
from app.services.ingestion_review import approve_document_for_indexing


@dataclass
class OcrApp:
    client: TestClient
    session_local: sessionmaker[Session]
    processed_root: Path


class FakeEmbeddingEncoder:
    model_name = "BAAI/bge-base-en-v1.5"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.5, 1.0] for index, _ in enumerate(texts)]


class FakeChromaCollection:
    name = "medical_record_chunks"

    def __init__(self) -> None:
        self.payload: dict | None = None

    def upsert(self, **payload) -> None:
        self.payload = payload


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
    original_openrouter_api_key = settings.openrouter_api_key
    original_openrouter_model = settings.openrouter_ocr_model
    original_openrouter_fallback_models = settings.openrouter_ocr_fallback_models
    settings.document_storage_root = tmp_path / "storage"
    settings.processed_text_root = tmp_path / "processed"
    settings.ocr_confidence_review_threshold = 0.80
    settings.openrouter_api_key = "test-openrouter-key"
    settings.openrouter_ocr_model = "baidu/qianfan-ocr-fast"
    settings.openrouter_ocr_fallback_models = "baidu/qianfan-ocr-fast:free"

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
    settings.openrouter_api_key = original_openrouter_api_key
    settings.openrouter_ocr_model = original_openrouter_model
    settings.openrouter_ocr_fallback_models = original_openrouter_fallback_models


def auth_header(ocr_app: OcrApp) -> dict[str, str]:
    response = ocr_app.client.post(
        "/api/v1/auth/login",
        json={"email": "records@example.com", "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def user_by_role(db: Session, role_name: RoleName) -> User:
    user = db.scalar(
        select(User).options(selectinload(User.roles)).where(User.roles.any(name=role_name))
    )
    assert user is not None
    return user


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


def mock_openrouter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str | list[str] = "Diagnosis: Hypertension\nMedication: Beta blockers",
    status_code: int = 200,
    error_body: dict | None = None,
) -> list[dict]:
    calls: list[dict] = []
    contents = [content] if isinstance(content, str) else content

    class FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, headers: dict, json: dict):
            response_content = contents[min(len(calls), len(contents) - 1)]
            calls.append({"url": url, "headers": headers, "json": json})
            if status_code >= 400:
                return httpx.Response(status_code, json=error_body or {"error": "failed"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": response_content}}]},
            )

    monkeypatch.setattr("app.services.document_ocr.httpx.Client", FakeClient)
    return calls


def mock_openrouter_sequence(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[httpx.Response],
) -> list[dict]:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url: str, headers: dict, json: dict):
            calls.append({"url": url, "headers": headers, "json": json})
            return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr("app.services.document_ocr.httpx.Client", FakeClient)
    return calls


def test_openrouter_payload_uses_enhanced_qianfan_system_prompt() -> None:
    payload = build_openrouter_ocr_payload(
        RenderedPage(page_number=3, image_data_url="data:image/png;base64,abc")
    )

    assert payload["model"] == "baidu/qianfan-ocr-fast"
    assert payload["temperature"] == 0
    assert payload["messages"][0]["content"] == QIANFAN_OCR_SYSTEM_PROMPT
    assert "Preserve the document's reading order" in QIANFAN_OCR_SYSTEM_PROMPT
    assert "Do not redact PHI" in QIANFAN_OCR_SYSTEM_PROMPT
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["text"].endswith("This is page 3.")
    assert user_content[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_scanned_pdf_qianfan_ocr_success_updates_document_and_job(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = mock_openrouter(monkeypatch)
    upload = upload_scanned_pdf(ocr_app)

    with ocr_app.session_local() as db:
        result = process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert calls
    assert result.engine == QIANFAN_OCR_ENGINE
    assert result.confidence == 1.0
    assert result.review_required is False
    assert "Diagnosis: Hypertension" in result.text
    assert "Medication: Beta blockers" in result.text

    processed_files = list(ocr_app.processed_root.glob("*.txt"))
    assert len(processed_files) == 1
    assert "Hypertension" in processed_files[0].read_text(encoding="utf-8")

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
    assert float(document.ocr_confidence) == 1.0
    assert document.ocr_engine == QIANFAN_OCR_ENGINE
    assert document.document_metadata["ocr"]["model"] == "baidu/qianfan-ocr-fast"
    assert document.document_metadata["ocr"]["engine_attempts"][-1] == {
        "engine": QIANFAN_OCR_ENGINE,
        "model": "baidu/qianfan-ocr-fast",
        "models": ["baidu/qianfan-ocr-fast"],
        "status": "succeeded",
        "page_count": 1,
        "max_workers": 1,
    }
    clinical = document.document_metadata["ocr"]["clinical"]
    assert clinical["diagnoses"] == ["Hypertension"]
    assert clinical["medications"] == ["Beta blockers"]
    chunking = document.document_metadata["ocr"]["chunking"]
    assert chunking["parent_chunk_count"] >= 1
    assert chunking["child_chunk_count"] >= 1
    assert len(chunks) >= 2
    assert any(chunk.parent_chunk_id is not None for chunk in chunks)
    assert all(float(chunk.ocr_confidence) == 1.0 for chunk in chunks)
    assert job is not None
    assert job.status.value == "succeeded"
    assert job.stage == "openrouter_ocr_extracted"


def test_qianfan_ocr_falls_back_when_primary_model_has_no_endpoint(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = mock_openrouter_sequence(
        monkeypatch,
        [
            httpx.Response(
                404,
                json={"error": {"message": "No endpoints found for baidu/qianfan-ocr-fast."}},
            ),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Diagnosis: Hypertension"}}]},
            ),
        ],
    )
    upload = upload_scanned_pdf(ocr_app)

    with ocr_app.session_local() as db:
        result = process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    assert [call["json"]["model"] for call in calls] == [
        "baidu/qianfan-ocr-fast",
        "baidu/qianfan-ocr-fast:free",
    ]
    assert "Hypertension" in result.text
    assert result.engine_attempts[-1]["model"] == "baidu/qianfan-ocr-fast:free"


def test_qianfan_reextract_reopens_review_and_approval_rebuilds_chunks(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_openrouter(
        monkeypatch,
        content=[
            "Diagnosis: Hypertension\nMedication: Beta blockers",
            "Diagnosis: Malaria\nMedication: artesunate",
        ],
    )
    upload = upload_scanned_pdf(ocr_app)
    collection = FakeChromaCollection()

    with ocr_app.session_local() as db:
        process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))
        approve_document_for_indexing(
            db,
            document_id=UUID(upload["id"]),
            approved_by=user_by_role(db, RoleName.RECORDS_STAFF),
            encoder=FakeEmbeddingEncoder(),
            collection=collection,
        )
        document = db.get(Document, UUID(upload["id"]))
        indexed_chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == UUID(upload["id"]))
            .order_by(DocumentChunk.chunk_index)
        ).all()

    assert document is not None
    assert document.status.value == "indexed"
    assert document.document_metadata["review_gate"]["status"] == "approved"
    assert indexed_chunks
    assert any("Hypertension" in chunk.content for chunk in indexed_chunks)

    with ocr_app.session_local() as db:
        result = process_ocr_ingestion_job(
            db,
            UUID(upload["ingestion_job_id"]),
            index_after_extraction=False,
            chunk_after_extraction=False,
        )
        document = db.get(Document, UUID(upload["id"]))
        chunks_after_reextract = db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == UUID(upload["id"]))
        ).all()

    assert "Malaria" in result.text
    assert document is not None
    assert document.status.value == "review_required"
    assert document.document_metadata["review_gate"]["status"] == "pending"
    assert document.document_metadata["extraction"]["review_gate"]["status"] == "pending"
    assert chunks_after_reextract == []

    detail_response = ocr_app.client.get(
        f"/api/v1/documents/{upload['id']}",
        headers=auth_header(ocr_app),
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["review_status"] == "pending"

    with ocr_app.session_local() as db:
        approval = approve_document_for_indexing(
            db,
            document_id=UUID(upload["id"]),
            approved_by=user_by_role(db, RoleName.RECORDS_STAFF),
            encoder=FakeEmbeddingEncoder(),
            collection=collection,
        )
        document = db.get(Document, UUID(upload["id"]))
        rebuilt_chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == UUID(upload["id"]))
            .order_by(DocumentChunk.chunk_index)
        ).all()

    assert approval.indexed_chunk_count == len(rebuilt_chunks)
    assert document is not None
    assert document.status.value == "indexed"
    assert document.document_metadata["review_gate"]["status"] == "approved"
    assert any("Malaria" in chunk.content for chunk in rebuilt_chunks)
    assert all("Hypertension" not in chunk.content for chunk in rebuilt_chunks)
    assert all(
        chunk.indexing_status == ChunkIndexingStatus.INDEXED.value for chunk in rebuilt_chunks
    )


def test_openrouter_ocr_errors_are_explicit_and_mark_job_failed(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_openrouter(
        monkeypatch,
        status_code=429,
        error_body={"error": {"message": "rate limit exceeded"}},
    )
    upload = upload_scanned_pdf(ocr_app)

    with ocr_app.session_local() as db:
        with pytest.raises(OcrError, match="HTTP 429"):
            process_ocr_ingestion_job(db, UUID(upload["ingestion_job_id"]))

    with ocr_app.session_local() as db:
        document = db.scalar(select(Document).where(Document.id == UUID(upload["id"])))
        job = db.scalar(
            select(IngestionJob).where(IngestionJob.id == UUID(upload["ingestion_job_id"]))
        )

    assert document is not None
    assert document.status.value == "failed"
    assert job is not None
    assert job.status.value == "failed"
    assert job.stage == "openrouter_ocr_failed"
    assert "HTTP 429" in (job.error_message or "")
    assert "rate limit exceeded" in (job.error_message or "")


def test_missing_openrouter_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", None)

    with pytest.raises(OcrError, match="OPENROUTER_API_KEY"):
        extract_scanned_pdf(build_pdf_bytes("Image-like scanned source"))


def test_queued_ocr_processor_processes_only_ocr_required_jobs(
    ocr_app: OcrApp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_openrouter(monkeypatch, content="Diagnosis: Malaria\nMedication: artesunate")
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
