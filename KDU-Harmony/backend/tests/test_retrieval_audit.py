from collections.abc import Generator
from typing import Any, cast
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import seed_auth_data
from app.models.audit_event import AuditEvent
from app.models.enums import AuditAction, RoleName
from app.models.user import User
from app.services.context_expansion import (
    ChunkContext,
    ContextualSearchHit,
    ContextualSearchResult,
    RetrievalConfidence,
    SourceCitation,
)
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


def build_context_result(db: Session, *, user: User) -> ContextualSearchResult:
    query = "patients with cardiac issues treated in Q1 2025"
    authorization = build_authorized_metadata_filter(db, user=user, query=query)
    document_id = UUID("41000000-0000-4000-8000-000000002401")
    chunk_id = UUID("51000000-0000-4000-8000-000000002401")
    return ContextualSearchResult(
        query=query,
        authorization=authorization,
        hits=[
            ContextualSearchHit(
                final_rank=1,
                matched_chunk=ChunkContext(
                    chunk_id=chunk_id,
                    section="Assessment",
                    text="Atrial fibrillation follow-up with metoprolol.",
                    page_number=4,
                    start_offset=10,
                    end_offset=58,
                    token_count=5,
                    chunk_type="child",
                ),
                parent_context=ChunkContext(
                    chunk_id=UUID("51000000-0000-4000-8000-000000002400"),
                    section="Assessment",
                    text="Assessment section parent context.",
                    page_number=4,
                    start_offset=0,
                    end_offset=90,
                    token_count=4,
                    chunk_type="parent",
                ),
                citation=SourceCitation(
                    document_id=document_id,
                    external_id="DOC-AUDIT-2401",
                    source_document="cardiology-note.pdf",
                    source_uri="local-encrypted://cardiology-note.pdf",
                    document_type="clinical_note",
                    page_number=4,
                    section="Assessment",
                    hospital="Harmony General Hospital",
                    physician="Dr. Asha Raman",
                    visit_id="VISIT-2401",
                    checksum_sha256="a" * 64,
                    citation_label="DOC-AUDIT-2401 | p. 4 | Assessment",
                    patient_ref="PATIENT_REF_2301",
                ),
                confidence=RetrievalConfidence(
                    score=0.89,
                    level="high",
                    reranker_score=0.86,
                    hybrid_score=0.032,
                    ocr_confidence=0.95,
                    source_count=2,
                ),
                retrieval={
                    "sources": ["bm25", "dense"],
                    "final_rank": 1,
                    "hybrid_rank": 1,
                    "hybrid_score": 0.032,
                    "reranker_score": 0.86,
                },
            )
        ],
        rerank_result=cast(Any, None),
    )


def test_rendered_retrieval_logs_query_and_document_access(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        doctor = user_by_role(db, RoleName.DOCTOR)
        doctor_id = doctor.id
        context_result = build_context_result(db, user=doctor)

        rendered = render_contextual_search_result(
            db,
            user=doctor,
            context_result=context_result,
            ip_address="127.0.0.1",
            user_agent="pytest",
        )

        events = list(db.scalars(select(AuditEvent).order_by(AuditEvent.action)).all())

    assert rendered.rendering_policy.render_mode == "full_phi"
    assert len(events) == 2

    query_event = next(event for event in events if event.action == AuditAction.QUERY_RUN)
    assert query_event.actor_user_id == doctor_id
    assert query_event.query_text == context_result.query
    assert query_event.role_snapshot == ["doctor"]
    assert query_event.decision == "allow"
    assert query_event.result_document_ids == ["41000000-0000-4000-8000-000000002401"]
    assert query_event.ip_address == "127.0.0.1"
    assert query_event.user_agent == "pytest"
    assert query_event.occurred_at is not None
    assert query_event.event_metadata["timestamp"]
    assert query_event.event_metadata["user_id"] == str(doctor_id)
    assert query_event.event_metadata["roles"] == ["doctor"]
    assert query_event.event_metadata["query"] == context_result.query
    assert query_event.event_metadata["document_ids"] == ["41000000-0000-4000-8000-000000002401"]
    assert query_event.event_metadata["chunk_ids"] == ["51000000-0000-4000-8000-000000002401"]
    assert query_event.event_metadata["masking_mode"] == "full_phi"
    assert query_event.event_metadata["masking_modes"] == ["full_phi"]
    assert query_event.event_metadata["patient_assignment_modes"] == ["assigned"]
    assert query_event.event_metadata["access_decision"] == "allow"
    assert query_event.event_metadata["filters"]["query_metadata_filters"][
        "diagnosis_category"
    ] == ["cardiac"]

    document_event = next(event for event in events if event.action == AuditAction.DOCUMENT_READ)
    assert document_event.actor_user_id == doctor_id
    assert document_event.resource_type == "document"
    assert str(document_event.resource_id) == "41000000-0000-4000-8000-000000002401"
    assert document_event.query_text == context_result.query
    assert document_event.role_snapshot == ["doctor"]
    assert document_event.decision == "allow"
    assert document_event.result_document_ids == ["41000000-0000-4000-8000-000000002401"]
    assert document_event.event_metadata["chunk_ids"] == ["51000000-0000-4000-8000-000000002401"]
    assert document_event.event_metadata["masking_mode"] == "full_phi"
    assert document_event.event_metadata["masking_modes"] == ["full_phi"]
    assert document_event.event_metadata["patient_assignment_modes"] == ["assigned"]
    assert document_event.event_metadata["access_decision"] == "allow"
    assert document_event.event_metadata["timestamp"] == query_event.event_metadata["timestamp"]


def test_denied_retrieval_logs_denied_query_without_document_access(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        user = User(email="nopolicy@example.com", display_name="No Policy", is_active=True)
        db.add(user)
        db.commit()
        db.refresh(user)
        authorization = build_authorized_metadata_filter(db, user=user, query="cardiac records")
        context_result = ContextualSearchResult(
            query="cardiac records",
            authorization=authorization,
            hits=[],
            rerank_result=cast(Any, None),
        )

        render_contextual_search_result(db, user=user, context_result=context_result)
        events = list(db.scalars(select(AuditEvent)).all())

    assert len(events) == 1
    assert events[0].action == AuditAction.QUERY_RUN
    assert events[0].decision == "deny"
    assert events[0].result_document_ids == []
    assert events[0].event_metadata["chunk_ids"] == []
    assert events[0].event_metadata["access_decision"] == "deny"
    assert events[0].event_metadata["filters"]["deny_reason"] == "no_active_document_access_policy"
