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
from app.services.bm25_retrieval import BM25SearchHit
from app.services.dense_retrieval import DenseSearchHit
from app.services.hybrid_retrieval import hybrid_search, reciprocal_rank_fusion


class FakeEmbeddingEncoder:
    model_name = "BAAI/bge-base-en-v1.5"

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeChromaCollection:
    name = "medical_record_chunks"

    def __init__(self, *, ids: list[str], distances: list[float] | None = None) -> None:
        self.ids = ids
        self.distances = distances or [0.1 for _ in ids]
        self.query_kwargs: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_kwargs = kwargs
        return {
            "ids": [self.ids],
            "distances": [self.distances],
            "metadatas": [[{"embedding_id": item} for item in self.ids]],
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
    diagnosis: str = "Hypertension",
    icd_codes: list[str] | None = None,
    medications: list[str] | None = None,
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM,
) -> DocumentChunk:
    icd_codes = icd_codes or ["I10"]
    if medications is None:
        medications = ["metoprolol 25 mg BID"]
    document = Document(
        id=document_id,
        external_id=f"DOC-{str(document_id)[-4:]}",
        patient_ref=patient_ref,
        visit_id=f"VISIT-{str(document_id)[-4:]}",
        document_type=DocumentType.CLINICAL_NOTE,
        status=DocumentStatus.INDEXED,
        file_name=f"{document_id}.txt",
        source_uri=f"local-encrypted://{document_id}",
        mime_type="text/plain",
        checksum_sha256="a" * 64,
        hospital="Harmony General Hospital",
        physician="Dr. Asha Raman",
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
            "document_type": document.document_type.value,
            "hospital": document.hospital,
            "physician": document.physician,
            "diagnosis": diagnosis,
            "icd_codes": icd_codes,
            "sensitivity_level": sensitivity_level.value,
            "clinical_entities": {
                "diagnoses": [diagnosis],
                "medications": medications,
                "symptoms": ["chest pressure"],
                "icd_codes": icd_codes,
                "dates": [{"label": "visit_date", "value": "2025-02-14"}],
            },
        },
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def test_rrf_promotes_chunks_found_by_both_retrievers() -> None:
    bm25_only = bm25_hit("51000000-0000-4000-8000-000000002001", score=9.0)
    shared_bm25 = bm25_hit("51000000-0000-4000-8000-000000002002", score=5.0)
    shared_dense = dense_hit("51000000-0000-4000-8000-000000002002", score=0.9)
    dense_only = dense_hit("51000000-0000-4000-8000-000000002003", score=0.8)

    hits = reciprocal_rank_fusion(
        bm25_hits=[bm25_only, shared_bm25],
        dense_hits=[shared_dense, dense_only],
        limit=3,
        rrf_k=60,
    )

    assert [str(hit.chunk_id) for hit in hits] == [
        "51000000-0000-4000-8000-000000002002",
        "51000000-0000-4000-8000-000000002001",
        "51000000-0000-4000-8000-000000002003",
    ]
    assert hits[0].sources == ["bm25", "dense"]
    assert hits[0].score == 0.03652247
    assert hits[0].source_metadata["fusion"]["source_overlap_boost"] == 0.004
    assert hits[0].bm25_score == 5.0
    assert hits[0].dense_score == 0.9


def test_hybrid_search_fuses_bm25_and_dense_results(
    session_local: sessionmaker[Session],
) -> None:
    encoder = FakeEmbeddingEncoder()

    with session_local() as db:
        shared = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002011"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002011"),
            patient_ref="PATIENT_REF_2011",
            content="Hypertension treatment includes metoprolol 25 mg BID.",
        )
        dense_only = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002012"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002012"),
            patient_ref="PATIENT_REF_2012",
            content="Hypertension follow-up focused on lifestyle counseling.",
            medications=[],
        )
        collection = FakeChromaCollection(
            ids=[dense_only.embedding_id or "", shared.embedding_id or ""],
            distances=[0.01, 0.2],
        )

        result = hybrid_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="metoprolol records",
            limit=5,
            candidate_limit=5,
            encoder=encoder,
            collection=collection,
        )

    assert collection.query_kwargs is not None
    assert collection.query_kwargs["n_results"] == 5
    assert [hit.chunk_id for hit in result.hits[:2]] == [shared.id, dense_only.id]
    assert result.hits[0].sources == ["bm25", "dense"]
    assert result.hits[0].source_metadata["bm25"]["matched_fields"]["medication_names"] == [
        "metoprolol"
    ]
    assert result.hits[1].sources == ["dense"]
    assert result.bm25_result.hits[0].chunk_id == shared.id
    assert result.dense_result.hits[0].chunk_id == dense_only.id


def bm25_hit(chunk_id: str, *, score: float) -> BM25SearchHit:
    parsed_chunk_id = UUID(chunk_id)
    return BM25SearchHit(
        chunk_id=parsed_chunk_id,
        document_id=document_id_for_chunk(parsed_chunk_id),
        patient_ref=f"PATIENT_REF_{chunk_id[-4:]}",
        section="Assessment",
        score=score,
        bm25_score=score,
        exact_match_score=0.0,
        matched_fields={},
        snippet=f"BM25 snippet {chunk_id}",
    )


def dense_hit(chunk_id: str, *, score: float) -> DenseSearchHit:
    parsed_chunk_id = UUID(chunk_id)
    return DenseSearchHit(
        chunk_id=parsed_chunk_id,
        document_id=document_id_for_chunk(parsed_chunk_id),
        patient_ref=f"PATIENT_REF_{chunk_id[-4:]}",
        section="Assessment",
        score=score,
        distance=1 - score,
        embedding_id=f"chunk:{chunk_id}",
        embedding_collection="medical_record_chunks",
        snippet=f"Dense snippet {chunk_id}",
        chroma_metadata={},
    )


def document_id_for_chunk(chunk_id: UUID) -> UUID:
    return UUID(str(chunk_id).replace("51000000", "41000000", 1))
