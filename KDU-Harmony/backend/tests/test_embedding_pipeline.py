import hashlib
from collections.abc import Generator
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentStatus, DocumentType, SensitivityLevel
from app.services.embedding_pipeline import (
    EmbeddingPipelineError,
    index_chunks_with_embeddings,
    index_pending_chunks,
)


class FakeEmbeddingEncoder:
    model_name = "BAAI/bge-base-en-v1.5"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [
            [float(index), float(index) + 0.1, float(index) + 0.2] for index, _ in enumerate(texts)
        ]


class BadEmbeddingEncoder:
    model_name = "BAAI/bge-base-en-v1.5"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0], [3.0]]


class FakeChromaCollection:
    name = "medical_record_chunks"

    def __init__(self) -> None:
        self.payload: dict | None = None

    def upsert(self, **payload) -> None:
        self.payload = payload


@pytest.fixture()
def session_local() -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    yield testing_session_local


def seed_document_with_chunks(db: Session) -> list[DocumentChunk]:
    document_id = UUID("41000000-0000-4000-8000-000000000001")
    parent_id = UUID("51000000-0000-4000-8000-000000000001")
    child_id = UUID("51000000-0000-4000-8000-000000000002")
    document = Document(
        id=document_id,
        external_id="DOC-EMBED-0001",
        patient_ref="PATIENT_REF_0142",
        visit_id="VISIT-0142",
        document_type=DocumentType.CLINICAL_NOTE,
        status=DocumentStatus.PROCESSED,
        file_name="note.txt",
        source_uri="local-encrypted://note",
        mime_type="text/plain",
        checksum_sha256="b" * 64,
        hospital="Harmony General Hospital",
        physician="Dr. Asha Raman",
        diagnosis="Hypertension",
        icd_codes=["I10"],
        sensitivity_level=SensitivityLevel.HIGH,
        is_encrypted=True,
        ocr_required=False,
        document_metadata={},
    )
    parent_content = "Diagnosis:\n\nHypertension with chest pressure."
    child_content = "Hypertension with chest pressure."
    parent_chunk = build_chunk(
        document=document,
        chunk_id=parent_id,
        chunk_index=0,
        parent_chunk_id=None,
        content=parent_content,
        chunk_type="parent",
    )
    child_chunk = build_chunk(
        document=document,
        chunk_id=child_id,
        chunk_index=1,
        parent_chunk_id=parent_id,
        content=child_content,
        chunk_type="child",
    )
    db.add(document)
    db.add_all([parent_chunk, child_chunk])
    db.commit()
    return [parent_chunk, child_chunk]


def build_chunk(
    *,
    document: Document,
    chunk_id: UUID,
    chunk_index: int,
    parent_chunk_id: UUID | None,
    content: str,
    chunk_type: str,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document=document,
        parent_chunk_id=parent_chunk_id,
        chunk_index=chunk_index,
        section="Diagnosis",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        token_count=len(content.split()),
        start_offset=0,
        end_offset=len(content),
        sensitivity_level=SensitivityLevel.HIGH,
        retrieval_metadata={
            "chunk_type": chunk_type,
            "section": "Diagnosis",
            "patient_ref": document.patient_ref,
            "visit_id": document.visit_id,
            "document_id": str(document.id),
            "external_id": document.external_id,
            "document_type": document.document_type.value,
            "hospital": document.hospital,
            "physician": document.physician,
            "diagnosis": document.diagnosis,
            "icd_codes": document.icd_codes,
            "sensitivity_level": document.sensitivity_level.value,
            "clinical_entities": {
                "diagnoses": ["Hypertension"],
                "medications": ["metoprolol 25 mg BID"],
                "symptoms": ["chest pressure"],
                "icd_codes": ["I10"],
                "dates": [{"label": "visit_date", "value": "2025-02-14"}],
            },
        },
    )


def test_index_pending_chunks_generates_embeddings_and_stores_chroma_metadata(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()
    collection = FakeChromaCollection()

    with session_local() as db:
        chunks = seed_document_with_chunks(db)
        result = index_pending_chunks(
            db,
            encoder=encoder,
            collection=collection,
            collection_name=collection.name,
        )
        persisted_chunks = db.scalars(
            select(DocumentChunk).order_by(DocumentChunk.chunk_index)
        ).all()

    assert result.indexed_chunk_count == 2
    assert result.embedding_dimension == 3
    assert result.model_name == "BAAI/bge-base-en-v1.5"
    assert result.embedding_ids == [f"chunk:{chunk.id}" for chunk in chunks]
    assert "Section: Diagnosis" in encoder.texts[0]
    assert "metoprolol 25 mg BID" in encoder.texts[0]
    assert collection.payload is not None
    assert collection.payload["ids"] == result.embedding_ids
    assert collection.payload["embeddings"] == [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]
    assert collection.payload["metadatas"][0]["embedding_model"] == "BAAI/bge-base-en-v1.5"
    assert collection.payload["metadatas"][0]["embedding_dimension"] == 3
    assert all(chunk.embedding_collection == collection.name for chunk in persisted_chunks)
    assert all(chunk.embedding_id for chunk in persisted_chunks)
    assert all(
        chunk.retrieval_metadata["embedding"]["model"] == "BAAI/bge-base-en-v1.5"
        for chunk in persisted_chunks
    )


def test_index_pending_chunks_skips_already_indexed_chunks(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()
    collection = FakeChromaCollection()

    with session_local() as db:
        seed_document_with_chunks(db)
        first_result = index_pending_chunks(db, encoder=encoder, collection=collection)
        second_result = index_pending_chunks(db, encoder=encoder, collection=collection)

    assert first_result.indexed_chunk_count == 2
    assert second_result.indexed_chunk_count == 0


def test_index_chunks_rejects_inconsistent_embedding_dimensions(
    session_local: sessionmaker[Session],
) -> None:
    collection = FakeChromaCollection()

    with session_local() as db:
        chunks = seed_document_with_chunks(db)
        with pytest.raises(EmbeddingPipelineError, match="consistent dimensions"):
            index_chunks_with_embeddings(
                db,
                chunks=chunks,
                encoder=BadEmbeddingEncoder(),
                collection=collection,
            )

    assert collection.payload is None
