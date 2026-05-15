import hashlib
from collections.abc import Generator
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import seed_auth_data
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import (
    ChunkIndexingStatus,
    DocumentStatus,
    DocumentType,
    RoleName,
    SensitivityLevel,
)
from app.models.user import User
from app.services.dense_retrieval import dense_vector_search


class FakeEmbeddingEncoder:
    model_name = "BAAI/bge-base-en-v1.5"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeChromaCollection:
    name = "medical_record_chunks"

    def __init__(
        self,
        *,
        ids: list[str],
        distances: list[float] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ids = ids
        self.distances = distances or [0.2 for _ in ids]
        self.metadatas = metadatas or [{} for _ in ids]
        self.query_kwargs: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_kwargs = kwargs
        return {
            "ids": [self.ids],
            "distances": [self.distances],
            "metadatas": [self.metadatas],
            "documents": [["" for _ in self.ids]],
        }


@pytest.fixture()
def session_local() -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    with testing_session_local() as db:
        seed_auth_data(db)
    yield testing_session_local


def user_by_role(db: Session, role_name: RoleName) -> User:
    user = db.scalar(
        select(User).options(selectinload(User.roles)).where(User.roles.any(name=role_name))
    )
    assert user is not None
    return user


def seed_indexed_chunk(
    db: Session,
    *,
    document_id: UUID,
    chunk_id: UUID,
    patient_ref: str,
    content: str,
    diagnosis: str,
    icd_codes: list[str],
    physician: str,
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM,
    hospital: str = "Harmony General Hospital",
    document_type: DocumentType = DocumentType.CLINICAL_NOTE,
    visit_date: str = "2025-02-14",
) -> DocumentChunk:
    document = Document(
        id=document_id,
        external_id=f"DOC-{str(document_id)[-4:]}",
        patient_ref=patient_ref,
        visit_id=f"VISIT-{str(document_id)[-4:]}",
        document_type=document_type,
        status=DocumentStatus.INDEXED,
        file_name=f"{document_id}.txt",
        source_uri=f"local-encrypted://{document_id}",
        mime_type="text/plain",
        checksum_sha256="a" * 64,
        hospital=hospital,
        physician=physician,
        diagnosis=diagnosis,
        icd_codes=icd_codes,
        sensitivity_level=sensitivity_level,
        is_encrypted=True,
        ocr_required=False,
        document_metadata={},
    )
    chunk = DocumentChunk(
        id=chunk_id,
        document=document,
        chunk_index=0,
        section="Assessment",
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        embedding_collection="medical_record_chunks",
        embedding_id=f"chunk:{chunk_id}",
        indexing_status=ChunkIndexingStatus.INDEXED.value,
        token_count=len(content.split()),
        start_offset=0,
        end_offset=len(content),
        sensitivity_level=sensitivity_level,
        retrieval_metadata={
            "chunk_type": "child",
            "section": "Assessment",
            "patient_ref": patient_ref,
            "visit_id": document.visit_id,
            "document_id": str(document_id),
            "external_id": document.external_id,
            "document_type": document_type.value,
            "hospital": hospital,
            "physician": physician,
            "diagnosis": diagnosis,
            "icd_codes": icd_codes,
            "sensitivity_level": sensitivity_level.value,
            "clinical_entities": {
                "diagnoses": [diagnosis],
                "medications": ["metoprolol 25 mg BID"],
                "symptoms": ["chest pressure"],
                "icd_codes": icd_codes,
                "dates": [{"label": "visit_date", "value": visit_date}],
            },
        },
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def test_dense_vector_search_embeds_query_and_hydrates_chroma_hits(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()

    with session_local() as db:
        expected = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000001901"),
            chunk_id=UUID("51000000-0000-4000-8000-000000001901"),
            patient_ref="PATIENT_REF_1901",
            content="Atrial fibrillation follow-up with chest pressure and metoprolol.",
            diagnosis="Atrial fibrillation",
            icd_codes=["I48.91"],
            physician="Dr. Asha Raman",
            sensitivity_level=SensitivityLevel.HIGH,
            visit_date="2025-02-14",
        )
        collection = FakeChromaCollection(
            ids=[expected.embedding_id or ""],
            distances=[0.2],
            metadatas=[{"patient_ref": "PATIENT_REF_1901"}],
        )

        result = dense_vector_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="patients with cardiac issues treated in Q1 2025",
            limit=5,
            encoder=encoder,
            collection=collection,
        )

    assert result.candidate_count == 1
    assert result.query_embedding_dimension == 3
    assert "Atrial fibrillation" in encoder.texts[0]
    assert "Visit Dates: 2025-01-01 to 2025-03-31" in encoder.texts[0]
    assert collection.query_kwargs is not None
    assert collection.query_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
    assert collection.query_kwargs["n_results"] == 250
    assert "where" in collection.query_kwargs
    query_predicates = flatten_and_predicates(collection.query_kwargs["where"])
    assert {"sensitivity_level": {"$in": ["high", "low", "medium", "restricted"]}} in (
        query_predicates
    )
    assert not any("visit_date" in predicate for predicate in query_predicates)
    assert result.hits[0].chunk_id == expected.id
    assert result.hits[0].score == 0.833333
    assert result.hits[0].chroma_metadata == {"patient_ref": "PATIENT_REF_1901"}


def test_dense_vector_search_does_not_query_chroma_when_authorization_denies(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()
    collection = FakeChromaCollection(ids=[])

    with session_local() as db:
        user = User(email="nopolicy@example.com", display_name="No Policy User", is_active=True)
        db.add(user)
        db.commit()
        db.refresh(user)

        result = dense_vector_search(
            db,
            user=user,
            query="cardiac records",
            encoder=encoder,
            collection=collection,
        )

    assert result.authorization.denied is True
    assert result.hits == []
    assert result.query_embedding_dimension == 0
    assert encoder.texts == []
    assert collection.query_kwargs is None


def test_dense_vector_search_rechecks_authorization_when_hydrating_hits(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()

    with session_local() as db:
        restricted = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000001902"),
            chunk_id=UUID("51000000-0000-4000-8000-000000001902"),
            patient_ref="PATIENT_REF_1902",
            content="Restricted hypertension note with metoprolol.",
            diagnosis="Hypertension",
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
            sensitivity_level=SensitivityLevel.RESTRICTED,
        )
        allowed = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000001903"),
            chunk_id=UUID("51000000-0000-4000-8000-000000001903"),
            patient_ref="PATIENT_REF_1903",
            content="De-identified hypertension follow-up with metoprolol.",
            diagnosis="Hypertension",
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
            sensitivity_level=SensitivityLevel.MEDIUM,
        )
        collection = FakeChromaCollection(
            ids=[restricted.embedding_id or "", allowed.embedding_id or ""],
            distances=[0.01, 0.2],
        )

        result = dense_vector_search(
            db,
            user=user_by_role(db, RoleName.RESEARCHER),
            query="hypertension records",
            encoder=encoder,
            collection=collection,
        )

    assert collection.query_kwargs is not None
    assert {"sensitivity_level": {"$in": ["low", "medium"]}} in flatten_and_predicates(
        collection.query_kwargs["where"]
    )
    assert [hit.chunk_id for hit in result.hits] == [allowed.id]


def flatten_and_predicates(where: dict[str, Any]) -> list[dict[str, Any]]:
    if "$and" in where:
        return where["$and"]
    return [where]
