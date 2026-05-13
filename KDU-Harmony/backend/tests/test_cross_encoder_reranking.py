import hashlib
import sys
from collections.abc import Generator
from types import SimpleNamespace
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
from app.services.cross_encoder_reranking import (
    SentenceTransformersCrossEncoderReranker,
    reranked_hybrid_search,
)


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


class FakeCrossEncoderReranker:
    model_name = "fake-cross-encoder"

    def __init__(self, score_by_phrase: dict[str, float] | None = None) -> None:
        self.score_by_phrase = score_by_phrase or {}
        self.pairs: list[tuple[str, str]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        scores: list[float] = []
        for _, document_text in pairs:
            score = 0.0
            for phrase, phrase_score in self.score_by_phrase.items():
                if phrase in document_text:
                    score = phrase_score
                    break
            scores.append(score)
        return scores


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
    medications: list[str] | None = None,
    sensitivity_level: SensitivityLevel = SensitivityLevel.MEDIUM,
) -> DocumentChunk:
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
        icd_codes=["I10"],
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
            "icd_codes": ["I10"],
            "sensitivity_level": sensitivity_level.value,
            "clinical_entities": {
                "diagnoses": [diagnosis],
                "medications": medications,
                "symptoms": ["chest pressure"],
                "icd_codes": ["I10"],
                "dates": [{"label": "visit_date", "value": "2025-02-14"}],
            },
        },
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def test_cross_encoder_reranks_top_hybrid_candidates(
    session_local: sessionmaker[Session],
) -> None:
    reranker = FakeCrossEncoderReranker(
        score_by_phrase={
            "lifestyle counseling": 0.95,
            "metoprolol 25 mg BID": 0.2,
        }
    )

    with session_local() as db:
        shared = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002101"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002101"),
            patient_ref="PATIENT_REF_2101",
            content="Hypertension treatment includes metoprolol 25 mg BID.",
        )
        dense_only = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002102"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002102"),
            patient_ref="PATIENT_REF_2102",
            content="Hypertension follow-up focused on lifestyle counseling.",
            medications=[],
        )
        collection = FakeChromaCollection(
            ids=[dense_only.embedding_id or "", shared.embedding_id or ""],
            distances=[0.01, 0.2],
        )

        result = reranked_hybrid_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="metoprolol records",
            limit=2,
            candidate_limit=5,
            rerank_top_n=2,
            encoder=FakeEmbeddingEncoder(),
            collection=collection,
            reranker=reranker,
        )

    assert result.hybrid_result.hits[0].chunk_id == shared.id
    assert [hit.chunk_id for hit in result.hits] == [dense_only.id, shared.id]
    assert result.hits[0].reranker_score == 0.95
    assert result.hits[0].hybrid_rank == 2
    assert result.hits[1].reranker_score == 0.2
    assert result.reranked_count == 2
    assert result.reranker_model == "fake-cross-encoder"
    assert len(reranker.pairs) == 2
    assert reranker.pairs[0][0].startswith("metoprolol records")
    assert "Hypertension treatment includes metoprolol" in reranker.pairs[0][1]


def test_rerank_top_n_limits_cross_encoder_work_and_preserves_tail_order(
    session_local: sessionmaker[Session],
) -> None:
    reranker = FakeCrossEncoderReranker(score_by_phrase={"metoprolol": 0.1})

    with session_local() as db:
        shared = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002111"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002111"),
            patient_ref="PATIENT_REF_2111",
            content="Hypertension treatment includes metoprolol 25 mg BID.",
        )
        dense_first = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002112"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002112"),
            patient_ref="PATIENT_REF_2112",
            content="Hypertension follow-up focused on lifestyle counseling.",
            medications=[],
        )
        dense_second = seed_indexed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000002113"),
            chunk_id=UUID("51000000-0000-4000-8000-000000002113"),
            patient_ref="PATIENT_REF_2113",
            content="Hypertension follow-up focused on sodium reduction.",
            medications=[],
        )
        collection = FakeChromaCollection(
            ids=[
                dense_first.embedding_id or "",
                dense_second.embedding_id or "",
                shared.embedding_id or "",
            ],
            distances=[0.01, 0.02, 0.2],
        )

        result = reranked_hybrid_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="metoprolol records",
            limit=3,
            candidate_limit=5,
            rerank_top_n=1,
            encoder=FakeEmbeddingEncoder(),
            collection=collection,
            reranker=reranker,
        )

    assert len(reranker.pairs) == 1
    assert [hit.chunk_id for hit in result.hits] == [shared.id, dense_first.id, dense_second.id]
    assert result.hits[0].reranker_score == 0.1
    assert result.hits[1].reranker_score is None
    assert result.hits[2].reranker_score is None
    assert result.reranked_count == 1


def test_sentence_transformers_reranker_falls_back_to_smaller_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_models: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            loaded_models.append(model_name)
            if model_name == "BAAI/bge-reranker-large":
                raise RuntimeError("large model unavailable")
            self.model_name = model_name

        def predict(
            self,
            pairs: list[tuple[str, str]],
            *,
            batch_size: int,
            show_progress_bar: bool,
        ) -> list[float]:
            assert batch_size > 0
            assert show_progress_bar is False
            return [float(index) for index, _ in enumerate(pairs)]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )

    reranker = SentenceTransformersCrossEncoderReranker(
        model_name="BAAI/bge-reranker-large",
        fallback_model_name="BAAI/bge-reranker-base",
    )

    assert loaded_models == ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base"]
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker.fallback_used is True
    assert reranker.score_pairs([("query", "first"), ("query", "second")]) == [0.0, 1.0]
