from collections.abc import Generator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import DEMO_PASSWORD, seed_auth_data
from app.db.session import get_db
from app.main import create_app
from app.models.audit_event import AuditEvent
from app.models.enums import AuditAction
from app.models.phi_mapping import PhiMapping
from app.services.phi_tokenization import tokenize_phi_text, upsert_phi_mappings


@dataclass
class PhiLookupApp:
    client: TestClient
    session_local: sessionmaker[Session]


@dataclass(frozen=True)
class StoredPhiFixture:
    patient_ref: str
    tokens_by_type: dict[str, str]


@pytest.fixture()
def phi_lookup_app() -> Generator[PhiLookupApp, None, None]:
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

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield PhiLookupApp(client=test_client, session_local=testing_session_local)
    app.dependency_overrides.clear()


def auth_header(phi_lookup_app: PhiLookupApp, email: str) -> dict[str, str]:
    response = phi_lookup_app.client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def store_phi_fixture(
    phi_lookup_app: PhiLookupApp,
    *,
    patient_ref: str = "PATIENT_REF_0300",
) -> StoredPhiFixture:
    text = (
        "Patient Name: Jane Doe DOB: 1981-04-05 MRN: MRN-909090 "
        "Phone: (555) 456-7890 Email: jane.doe@example.com "
        "Address: 45 Cedar Street, Denver, CO 80202 Diagnosis: Hypertension"
    )
    result = tokenize_phi_text(text, patient_ref=patient_ref)
    tokens_by_type = {detection.entity_type: detection.token for detection in result.detections}

    with phi_lookup_app.session_local() as db:
        upsert_phi_mappings(
            db,
            detections=result.detections,
            patient_ref=patient_ref,
            created_by_user_id=None,
        )
        db.commit()

    return StoredPhiFixture(patient_ref=patient_ref, tokens_by_type=tokens_by_type)


def phi_audit_events(db: Session) -> list[AuditEvent]:
    return db.scalars(
        select(AuditEvent)
        .where(AuditEvent.action == AuditAction.PHI_DECRYPT)
        .order_by(AuditEvent.occurred_at)
    ).all()


def test_doctor_can_lookup_decrypted_phi_value(phi_lookup_app: PhiLookupApp) -> None:
    stored_phi = store_phi_fixture(phi_lookup_app)
    patient_name_token = stored_phi.tokens_by_type["PATIENT_NAME"]

    response = phi_lookup_app.client.post(
        "/api/v1/phi/lookup",
        headers=auth_header(phi_lookup_app, "doctor@example.com"),
        json={"token": patient_name_token, "patient_ref": stored_phi.patient_ref},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token"] == patient_name_token
    assert body["patient_ref"] == stored_phi.patient_ref
    assert body["entity_type"] == "PATIENT_NAME"
    assert body["value"] == "Jane Doe"

    with phi_lookup_app.session_local() as db:
        mapping = db.scalar(select(PhiMapping).where(PhiMapping.token == patient_name_token))
        audits = phi_audit_events(db)

    assert mapping is not None
    assert mapping.last_accessed_at is not None
    assert len(audits) == 1
    assert audits[0].decision == "allow"
    assert audits[0].event_metadata["reason"] == "phi_visibility_allowed"


def test_records_staff_can_lookup_operational_phi_value(
    phi_lookup_app: PhiLookupApp,
) -> None:
    stored_phi = store_phi_fixture(phi_lookup_app)
    mrn_token = stored_phi.tokens_by_type["MRN"]

    response = phi_lookup_app.client.post(
        "/api/v1/phi/lookup",
        headers=auth_header(phi_lookup_app, "records@example.com"),
        json={"token": mrn_token, "patient_ref": stored_phi.patient_ref},
    )

    assert response.status_code == 200
    assert response.json()["entity_type"] == "MRN"
    assert response.json()["value"] == "MRN-909090"


def test_doctor_cannot_lookup_unassigned_patient_phi_value(phi_lookup_app: PhiLookupApp) -> None:
    stored_phi = store_phi_fixture(phi_lookup_app, patient_ref="PATIENT_REF_UNASSIGNED")
    patient_name_token = stored_phi.tokens_by_type["PATIENT_NAME"]

    response = phi_lookup_app.client.post(
        "/api/v1/phi/lookup",
        headers=auth_header(phi_lookup_app, "doctor@example.com"),
        json={"token": patient_name_token, "patient_ref": stored_phi.patient_ref},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "PHI lookup is not permitted for this user"

    with phi_lookup_app.session_local() as db:
        audits = phi_audit_events(db)

    assert len(audits) == 1
    assert audits[0].decision == "deny"
    assert audits[0].event_metadata["reason"] == "patient_assignment_required"


def test_researcher_cannot_lookup_decrypted_phi_value(phi_lookup_app: PhiLookupApp) -> None:
    stored_phi = store_phi_fixture(phi_lookup_app)
    email_token = stored_phi.tokens_by_type["EMAIL"]

    response = phi_lookup_app.client.post(
        "/api/v1/phi/lookup",
        headers=auth_header(phi_lookup_app, "researcher@example.com"),
        json={"token": email_token, "patient_ref": stored_phi.patient_ref},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "PHI lookup is not permitted for this user"

    with phi_lookup_app.session_local() as db:
        mapping = db.scalar(select(PhiMapping).where(PhiMapping.token == email_token))
        audits = phi_audit_events(db)

    assert mapping is not None
    assert mapping.last_accessed_at is None
    assert len(audits) == 1
    assert audits[0].decision == "deny"
    assert audits[0].event_metadata["reason"] == "phi_visibility_denied"


def test_phi_lookup_patient_ref_mismatch_is_not_found(phi_lookup_app: PhiLookupApp) -> None:
    stored_phi = store_phi_fixture(phi_lookup_app)
    dob_token = stored_phi.tokens_by_type["DOB"]

    response = phi_lookup_app.client.post(
        "/api/v1/phi/lookup",
        headers=auth_header(phi_lookup_app, "doctor@example.com"),
        json={"token": dob_token, "patient_ref": "PATIENT_REF_OTHER"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "PHI token was not found"

    with phi_lookup_app.session_local() as db:
        audits = phi_audit_events(db)

    assert len(audits) == 1
    assert audits[0].decision == "not_found"
    assert audits[0].resource_id is None
