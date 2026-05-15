import hashlib
from collections.abc import Generator
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import ChunkIndexingStatus, DocumentStatus, DocumentType, SensitivityLevel
from app.services.context_expansion import expand_reranked_hits
from app.services.cross_encoder_reranking import RerankedSearchHit


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


def seed_parent_child_chunks(db: Session) -> tuple[DocumentChunk, DocumentChunk]:
    document = Document(
        id=UUID("41000000-0000-4000-8000-000000002201"),
        external_id="DOC-CTX-2201",
        patient_ref="PATIENT_REF_2201",
        visit_id="VISIT-2201",
        document_type=DocumentType.TYPED_PDF,
        status=DocumentStatus.INDEXED,
        file_name="cardiology-note.pdf",
        source_uri="local-encrypted://cardiology-note.pdf",
        mime_type="application/pdf",
        checksum_sha256="a" * 64,
        hospital="Harmony General Hospital",
        physician="Dr. Asha Raman",
        diagnosis="Atrial fibrillation",
        icd_codes=["I48.91"],
        sensitivity_level=SensitivityLevel.HIGH,
        is_encrypted=True,
        ocr_required=False,
        ocr_confidence=Decimal("0.9100"),
        document_metadata={},
    )
    parent_content = (
        "Assessment:\nAtrial fibrillation with palpitations. "
        "Treatment plan includes metoprolol and cardiology follow-up."
    )
    child_content = "Atrial fibrillation with palpitations."
    parent = DocumentChunk(
        id=UUID("51000000-0000-4000-8000-000000002201"),
        document=document,
        chunk_index=0,
        section="Assessment",
        content=parent_content,
        content_sha256=hashlib.sha256(parent_content.encode("utf-8")).hexdigest(),
        embedding_collection="medical_record_chunks",
        embedding_id=f"chunk:{UUID('51000000-0000-4000-8000-000000002201')}",
        indexing_status=ChunkIndexingStatus.INDEXED.value,
        token_count=len(parent_content.split()),
        start_offset=10,
        end_offset=10 + len(parent_content),
        page_number=3,
        ocr_confidence=Decimal("0.9200"),
        sensitivity_level=SensitivityLevel.HIGH,
        retrieval_metadata={
            "chunk_type": "parent",
            "section": "Assessment",
            "patient_ref": document.patient_ref,
            "document_id": str(document.id),
            "external_id": document.external_id,
            "document_type": document.document_type.value,
            "hospital": document.hospital,
            "physician": document.physician,
            "diagnosis": document.diagnosis,
            "icd_codes": document.icd_codes,
            "sensitivity_level": document.sensitivity_level.value,
        },
    )
    child = DocumentChunk(
        id=UUID("51000000-0000-4000-8000-000000002202"),
        document=document,
        parent_chunk=parent,
        chunk_index=1,
        section="Assessment",
        content=child_content,
        content_sha256=hashlib.sha256(child_content.encode("utf-8")).hexdigest(),
        embedding_collection="medical_record_chunks",
        embedding_id=f"chunk:{UUID('51000000-0000-4000-8000-000000002202')}",
        indexing_status=ChunkIndexingStatus.INDEXED.value,
        token_count=len(child_content.split()),
        start_offset=22,
        end_offset=22 + len(child_content),
        page_number=3,
        ocr_confidence=Decimal("0.9200"),
        sensitivity_level=SensitivityLevel.HIGH,
        retrieval_metadata={
            "chunk_type": "child",
            "section": "Assessment",
            "parent_chunk_id": str(parent.id),
            "patient_ref": document.patient_ref,
            "document_id": str(document.id),
            "external_id": document.external_id,
            "document_type": document.document_type.value,
            "hospital": document.hospital,
            "physician": document.physician,
            "diagnosis": document.diagnosis,
            "icd_codes": document.icd_codes,
            "sensitivity_level": document.sensitivity_level.value,
        },
    )
    db.add_all([parent, child])
    db.commit()
    db.refresh(parent)
    db.refresh(child)
    return parent, child


def test_expand_reranked_hit_returns_parent_context_and_citation(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        parent, child = seed_parent_child_chunks(db)
        hits = expand_reranked_hits(
            db,
            hits=[
                RerankedSearchHit(
                    chunk_id=child.id,
                    document_id=child.document_id,
                    patient_ref="PATIENT_REF_2201",
                    section="Assessment",
                    final_rank=1,
                    hybrid_rank=2,
                    hybrid_score=0.03252247,
                    reranker_score=0.95,
                    sources=["bm25", "dense"],
                    snippet=child.content,
                    source_metadata={"dense": {"distance": 0.2}},
                )
            ],
        )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.matched_chunk.chunk_id == child.id
    assert hit.matched_chunk.text == "Atrial fibrillation with palpitations."
    assert hit.parent_context is not None
    assert hit.parent_context.chunk_id == parent.id
    assert "Treatment plan includes metoprolol" in hit.parent_context.text
    assert hit.citation.document_id == child.document_id
    assert hit.citation.external_id == "DOC-CTX-2201"
    assert hit.citation.source_document == "cardiology-note.pdf"
    assert hit.citation.page_number == 3
    assert hit.citation.section == "Assessment"
    assert hit.citation.citation_label == "DOC-CTX-2201 | p. 3 | Assessment"
    assert hit.confidence.level == "high"
    assert hit.confidence.score == 0.9455
    assert hit.confidence.ocr_confidence == 0.92
    assert hit.retrieval["sources"] == ["bm25", "dense"]


def test_expand_parent_match_uses_parent_as_context(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        parent, _ = seed_parent_child_chunks(db)
        hits = expand_reranked_hits(
            db,
            hits=[
                RerankedSearchHit(
                    chunk_id=parent.id,
                    document_id=parent.document_id,
                    patient_ref="PATIENT_REF_2201",
                    section="Assessment",
                    final_rank=1,
                    hybrid_rank=1,
                    hybrid_score=0.025,
                    reranker_score=None,
                    sources=["dense"],
                    snippet=parent.content[:80],
                    source_metadata={},
                )
            ],
        )

    assert len(hits) == 1
    assert hits[0].matched_chunk.chunk_id == parent.id
    assert hits[0].parent_context is not None
    assert hits[0].parent_context.chunk_id == parent.id
    assert hits[0].confidence.level == "medium"


def test_expand_reranked_hits_deduplicates_same_document_section_text(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        parent, child = seed_parent_child_chunks(db)
        duplicate_child = DocumentChunk(
            id=UUID("51000000-0000-4000-8000-000000002299"),
            document=child.document,
            parent_chunk=parent,
            chunk_index=2,
            section=child.section,
            content=child.content,
            content_sha256=child.content_sha256,
            embedding_collection="medical_record_chunks",
            embedding_id=f"chunk:{UUID('51000000-0000-4000-8000-000000002299')}",
            indexing_status=ChunkIndexingStatus.INDEXED.value,
            token_count=child.token_count,
            start_offset=child.end_offset or 0,
            end_offset=(child.end_offset or 0) + len(child.content),
            page_number=child.page_number,
            ocr_confidence=child.ocr_confidence,
            sensitivity_level=child.sensitivity_level,
            retrieval_metadata=child.retrieval_metadata,
        )
        db.add(duplicate_child)
        db.commit()
        hits = expand_reranked_hits(
            db,
            hits=[
                RerankedSearchHit(
                    chunk_id=child.id,
                    document_id=child.document_id,
                    patient_ref="PATIENT_REF_2201",
                    section="Assessment",
                    final_rank=1,
                    hybrid_rank=1,
                    hybrid_score=0.04,
                    reranker_score=0.98,
                    sources=["bm25", "dense"],
                    snippet=child.content,
                    source_metadata={},
                ),
                RerankedSearchHit(
                    chunk_id=duplicate_child.id,
                    document_id=duplicate_child.document_id,
                    patient_ref="PATIENT_REF_2201",
                    section="Assessment",
                    final_rank=2,
                    hybrid_rank=2,
                    hybrid_score=0.03,
                    reranker_score=0.96,
                    sources=["bm25"],
                    snippet=duplicate_child.content,
                    source_metadata={},
                ),
            ],
        )

    assert len(hits) == 1
    assert hits[0].final_rank == 1
    assert hits[0].matched_chunk.chunk_id == child.id
