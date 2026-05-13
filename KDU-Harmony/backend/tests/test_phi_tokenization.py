from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.models.phi_mapping import PhiMapping
from app.services.phi_tokenization import (
    decrypt_phi_value,
    tokenize_phi_text,
    upsert_phi_mappings,
)


def test_tokenize_phi_text_replaces_direct_identifiers() -> None:
    text = (
        "Patient Name: John Smith DOB: 01/23/1980 MRN: MRN-442211 "
        "Phone: (555) 123-4567 Email: john.smith@example.com "
        "Address: 12 Oak Street, Boston, MA 02118 Diagnosis: Hypertension"
    )

    result = tokenize_phi_text(text, patient_ref="PATIENT_REF_0042")

    assert "John Smith" not in result.text
    assert "01/23/1980" not in result.text
    assert "MRN-442211" not in result.text
    assert "(555) 123-4567" not in result.text
    assert "john.smith@example.com" not in result.text
    assert "12 Oak Street" not in result.text
    assert "Patient Name: [PATIENT_REF_0042]" in result.text
    assert "DOB: [DOB_0042_" in result.text
    assert "MRN: [MRN_0042_" in result.text
    assert "Phone: [PHONE_0042_" in result.text
    assert "Email: [EMAIL_0042_" in result.text
    assert "Address: [ADDR_0042_" in result.text
    assert result.entity_counts == {
        "PATIENT_NAME": 1,
        "DOB": 1,
        "MRN": 1,
        "PHONE": 1,
        "EMAIL": 1,
        "ADDRESS": 1,
    }


def test_upsert_phi_mappings_encrypts_values_and_reuses_tokens() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=engine)

    text = "Patient Name: Jane Doe DOB: 1981-04-05 Email: jane.doe@example.com"
    result = tokenize_phi_text(text, patient_ref="PATIENT_REF_0043")

    with testing_session_local() as db:
        mappings = persist_mappings(db, result)
        mappings_again = persist_mappings(db, result)
        stored_count = len(db.scalars(select(PhiMapping)).all())

    assert len(mappings) == 3
    assert len(mappings_again) == 3
    assert stored_count == 3
    patient_mapping = next(mapping for mapping in mappings if mapping.token == "[PATIENT_REF_0043]")
    assert "Jane Doe" not in patient_mapping.encrypted_value
    assert (
        decrypt_phi_value(
            patient_mapping.encrypted_value,
            token=patient_mapping.token,
        )
        == "Jane Doe"
    )


def persist_mappings(db: Session, result):
    mappings = upsert_phi_mappings(
        db,
        detections=result.detections,
        patient_ref="PATIENT_REF_0043",
        created_by_user_id=None,
    )
    db.commit()
    return mappings
