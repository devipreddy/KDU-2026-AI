from collections.abc import Generator
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import seed_auth_data
from app.models.enums import RoleName
from app.models.phi_mapping import PhiMapping
from app.models.user import User
from app.services.context_expansion import (
    ChunkContext,
    ContextualSearchHit,
    ContextualSearchResult,
    RetrievalConfidence,
    SourceCitation,
)
from app.services.phi_tokenization import encrypt_phi_value, token_for_phi
from app.services.result_rendering import render_contextual_search_result
from app.services.retrieval_authorization import build_authorized_metadata_filter


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


def seed_phi_mapping(
    db: Session,
    *,
    patient_ref: str,
    entity_type: str,
    value: str,
) -> str:
    token = token_for_phi(entity_type=entity_type, value=value, patient_ref=patient_ref)
    db.add(
        PhiMapping(
            patient_ref=patient_ref,
            token=token,
            entity_type=entity_type,
            encrypted_value=encrypt_phi_value(value, token=token),
            encryption_key_id="local-development-key",
        )
    )
    db.commit()
    return token


def build_context_result(db: Session, *, user: User, text: str) -> ContextualSearchResult:
    authorization = build_authorized_metadata_filter(db, user=user, query="cardiac records")
    document_id = UUID("41000000-0000-4000-8000-000000002301")
    chunk_id = UUID("51000000-0000-4000-8000-000000002301")
    hit = ContextualSearchHit(
        final_rank=1,
        matched_chunk=ChunkContext(
            chunk_id=chunk_id,
            section="Assessment",
            text=text,
            page_number=2,
            start_offset=10,
            end_offset=10 + len(text),
            token_count=len(text.split()),
            chunk_type="child",
        ),
        parent_context=ChunkContext(
            chunk_id=UUID("51000000-0000-4000-8000-000000002300"),
            section="Assessment",
            text=f"Assessment:\n{text}\nPlan: Continue metoprolol.",
            page_number=2,
            start_offset=0,
            end_offset=120,
            token_count=12,
            chunk_type="parent",
        ),
        citation=SourceCitation(
            document_id=document_id,
            external_id="DOC-PHI-2301",
            source_document="cardiology-note.pdf",
            source_uri="local-encrypted://cardiology-note.pdf",
            document_type="clinical_note",
            page_number=2,
            section="Assessment",
            hospital="Harmony General Hospital",
            physician="Dr. Asha Raman",
            visit_id="VISIT-2301",
            checksum_sha256="a" * 64,
            citation_label="DOC-PHI-2301 | p. 2 | Assessment",
        ),
        confidence=RetrievalConfidence(
            score=0.91,
            level="high",
            reranker_score=0.9,
            hybrid_score=0.032,
            ocr_confidence=0.95,
            source_count=2,
        ),
        retrieval={
            "sources": ["bm25", "dense"],
            "final_rank": 1,
            "hybrid_rank": 1,
            "hybrid_score": 0.032,
            "reranker_score": 0.9,
            "source_metadata": {
                "bm25": {"matched_fields": {"patient": ["[PATIENT_REF_2301]"]}},
                "dense": {"chroma_metadata": {"patient_ref": "PATIENT_REF_2301"}},
            },
        },
    )
    return ContextualSearchResult(
        query="cardiac records",
        authorization=authorization,
        hits=[hit],
        rerank_result=cast(Any, None),
    )


def test_doctor_rendering_decrypts_authorized_phi(
    session_local: sessionmaker[Session],
) -> None:
    patient_ref = "PATIENT_REF_2301"
    with session_local() as db:
        name_token = seed_phi_mapping(
            db,
            patient_ref=patient_ref,
            entity_type="PATIENT_NAME",
            value="John Smith",
        )
        dob_token = seed_phi_mapping(
            db,
            patient_ref=patient_ref,
            entity_type="DOB",
            value="1972-04-08",
        )
        doctor = user_by_role(db, RoleName.DOCTOR)
        context_result = build_context_result(
            db,
            user=doctor,
            text=f"Patient {name_token} DOB {dob_token} has atrial fibrillation.",
        )

        rendered = render_contextual_search_result(db, user=doctor, context_result=context_result)

    assert rendered.rendering_policy.render_mode == "full_phi"
    assert rendered.hits[0].matched_chunk.text is not None
    assert "John Smith" in rendered.hits[0].matched_chunk.text
    assert "1972-04-08" in rendered.hits[0].matched_chunk.text
    assert name_token not in rendered.hits[0].parent_context.text
    assert rendered.hits[0].redactions == []


def test_researcher_rendering_deidentifies_phi_and_keeps_clinical_text(
    session_local: sessionmaker[Session],
) -> None:
    patient_ref = "PATIENT_REF_2301"
    with session_local() as db:
        name_token = seed_phi_mapping(
            db,
            patient_ref=patient_ref,
            entity_type="PATIENT_NAME",
            value="John Smith",
        )
        researcher = user_by_role(db, RoleName.RESEARCHER)
        context_result = build_context_result(
            db,
            user=researcher,
            text=f"Patient {name_token} has atrial fibrillation treated with metoprolol.",
        )

        rendered = render_contextual_search_result(
            db,
            user=researcher,
            context_result=context_result,
        )

    hit = rendered.hits[0]
    assert rendered.rendering_policy.render_mode == "de_identified"
    assert hit.matched_chunk.text is not None
    assert "[DEID_PATIENT]" in hit.matched_chunk.text
    assert "John Smith" not in hit.matched_chunk.text
    assert name_token not in hit.matched_chunk.text
    assert "atrial fibrillation treated with metoprolol" in hit.matched_chunk.text
    assert hit.patient_display_ref is not None
    assert hit.patient_display_ref.startswith("DEID-")
    assert (
        hit.retrieval["source_metadata"]["dense"]["chroma_metadata"]["patient_ref"]
        == "[DEID_PATIENT]"
    )
    assert hit.redactions == ["PATIENT_NAME"]


def test_admin_rendering_returns_metadata_only_without_clinical_text(
    session_local: sessionmaker[Session],
) -> None:
    patient_ref = "PATIENT_REF_2301"
    with session_local() as db:
        name_token = seed_phi_mapping(
            db,
            patient_ref=patient_ref,
            entity_type="PATIENT_NAME",
            value="John Smith",
        )
        admin = user_by_role(db, RoleName.ADMIN)
        context_result = build_context_result(
            db,
            user=admin,
            text=f"Patient {name_token} has atrial fibrillation.",
        )

        rendered = render_contextual_search_result(db, user=admin, context_result=context_result)

    hit = rendered.hits[0]
    assert rendered.rendering_policy.render_mode == "metadata_only"
    assert hit.matched_chunk.text is None
    assert hit.parent_context is not None
    assert hit.parent_context.text is None
    assert hit.patient_display_ref is None
    assert hit.citation.source_document == "cardiology-note.pdf"
    assert hit.citation.page_number == 2
    assert hit.retrieval == {
        "sources": ["bm25", "dense"],
        "final_rank": 1,
        "hybrid_rank": 1,
        "hybrid_score": 0.032,
        "reranker_score": 0.9,
    }
