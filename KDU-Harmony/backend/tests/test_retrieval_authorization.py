from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import seed_auth_data
from app.models.access_policy import AccessPolicy
from app.models.enums import AccessPolicyEffect, RoleName
from app.models.user import User
from app.services.retrieval_authorization import (
    DENY_ALL_WHERE,
    build_authorized_metadata_filter,
)


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
        select(User)
        .options(selectinload(User.roles))
        .join(User.roles)
        .where(User.roles.any(name=role_name))
    )
    assert user is not None
    return user


def test_doctor_filter_combines_authorization_and_query_metadata(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="patients with cardiac issues treated in Q1 2025",
        )

    assert result.denied is False
    assert result.allowed_sensitivity_levels == ["low", "medium", "high", "restricted"]
    assert result.phi_visibility == "full"
    assert result.authorization_filters["sensitivity_level"] == [
        "low",
        "medium",
        "high",
        "restricted",
    ]
    assert result.query_metadata_filters["diagnosis_category"] == ["cardiac"]
    assert result.unmet_policy_requirements == ["requires_treatment_relationship"]

    predicates = flatten_and_predicates(result.chroma_where)
    assert {"hospital": {"$eq": "Harmony General Hospital"}} in predicates
    assert {"sensitivity_level": {"$in": ["high", "low", "medium", "restricted"]}} in predicates
    assert {"visit_date": {"$gte": "2025-01-01"}} in predicates
    assert {"visit_date": {"$lte": "2025-03-31"}} in predicates
    diagnosis_predicate = next(predicate for predicate in predicates if "diagnosis" in predicate)
    assert "Congestive heart failure" in diagnosis_predicate["diagnosis"]["$in"]
    assert "Stable angina" in diagnosis_predicate["diagnosis"]["$in"]


def test_researcher_filter_excludes_high_and_restricted_chunks(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.RESEARCHER),
            query="records from Harmony General Hospital",
        )

    assert result.denied is False
    assert result.allowed_sensitivity_levels == ["low", "medium"]
    assert result.phi_visibility == "de_identified"
    predicates = flatten_and_predicates(result.chroma_where)
    assert {"sensitivity_level": {"$in": ["low", "medium"]}} in predicates
    assert {"hospital": {"$eq": "Harmony General Hospital"}} in predicates


def test_doctor_query_outside_organization_scope_denies_retrieval(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="records from Mercy West Health",
        )

    assert result.denied is True
    assert result.deny_reason == "query_outside_authorized_scope"
    assert result.authorization_filters["hospital"] == ["Harmony General Hospital"]
    assert result.chroma_where == DENY_ALL_WHERE


def test_patient_scope_is_applied_before_retrieval(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.NURSE),
            query="patient ref 42 lab reports",
            authorized_patient_refs=["PATIENT_REF_0042"],
        )

    assert result.denied is False
    assert result.unmet_policy_requirements == []
    predicates = flatten_and_predicates(result.chroma_where)
    assert {"patient_ref": {"$eq": "PATIENT_REF_0042"}} in predicates
    assert {"document_type": {"$eq": "lab_report"}} in predicates
    assert {"sensitivity_level": {"$in": ["high", "low", "medium"]}} in predicates


def test_deny_policy_removes_sensitivity_from_authorized_scope(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        doctor = user_by_role(db, RoleName.DOCTOR)
        db.add(
            AccessPolicy(
                role_id=doctor.roles[0].id,
                name="deny_restricted_documents",
                effect=AccessPolicyEffect.DENY,
                resource_type="document",
                conditions={"sensitivity_levels": ["restricted"]},
                priority=10,
                is_active=True,
            )
        )
        db.commit()

        result = build_authorized_metadata_filter(db, user=doctor, query="cardiac records")

    assert result.allowed_sensitivity_levels == ["low", "medium", "high"]
    predicates = flatten_and_predicates(result.chroma_where)
    assert {"sensitivity_level": {"$in": ["high", "low", "medium"]}} in predicates


def test_conflicting_patient_scope_denies_retrieval(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.NURSE),
            query="patient ref 43 lab reports",
            authorized_patient_refs=["PATIENT_REF_0042"],
        )

    assert result.denied is True
    assert result.deny_reason == "query_outside_authorized_scope"
    assert result.chroma_where == DENY_ALL_WHERE


def test_user_without_document_policy_denies_all(session_local: sessionmaker[Session]) -> None:
    with session_local() as db:
        user = User(
            email="nopolicy@example.com",
            display_name="No Policy User",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        result = build_authorized_metadata_filter(db, user=user, query="cardiac records")

    assert result.denied is True
    assert result.deny_reason == "no_active_document_access_policy"
    assert result.chroma_where == DENY_ALL_WHERE


def test_exact_diagnosis_query_is_not_broadened_by_category_alias(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        result = build_authorized_metadata_filter(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="patients with chronic kidney disease during 2025",
        )

    predicates = flatten_and_predicates(result.chroma_where)
    diagnosis_predicate = next(predicate for predicate in predicates if "diagnosis" in predicate)
    assert diagnosis_predicate == {"diagnosis": {"$eq": "Chronic kidney disease stage 3"}}


def flatten_and_predicates(where: dict) -> list[dict]:
    if "$and" in where:
        return where["$and"]
    return [where]
