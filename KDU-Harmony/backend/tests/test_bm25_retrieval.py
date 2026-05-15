import hashlib
from collections.abc import Generator
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.seed import seed_auth_data
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentStatus, DocumentType, RoleName, SensitivityLevel
from app.models.phi_mapping import PhiMapping
from app.models.user import User
from app.services.bm25_retrieval import bm25_search
from app.services.phi_tokenization import encrypt_phi_value, token_for_phi


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


def seed_chunk(
    db: Session,
    *,
    document_id: UUID,
    chunk_id: UUID,
    patient_ref: str,
    content: str,
    diagnosis: str,
    medications: list[str],
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
                "medications": medications,
                "symptoms": [],
                "icd_codes": icd_codes,
                "dates": [{"label": "visit_date", "value": visit_date}],
            },
        },
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def test_bm25_retrieves_exact_medication_and_diagnosis_terms(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        expected = seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000101"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000101"),
            patient_ref="PATIENT_REF_0101",
            content="Diagnosis: Type 2 diabetes mellitus. Medication: metformin 500 mg BID.",
            diagnosis="Type 2 diabetes mellitus",
            medications=["metformin 500 mg BID"],
            icd_codes=["E11.9"],
            physician="Dr. Priya Nair",
        )
        seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000102"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000102"),
            patient_ref="PATIENT_REF_0102",
            content="Diagnosis: Hypertension. Medication: aspirin 81 mg daily.",
            diagnosis="Hypertension",
            medications=["aspirin 81 mg daily"],
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="diabetes patients prescribed metformin",
        )

    assert result.candidate_chunk_count == 1
    assert result.hits[0].chunk_id == expected.id
    assert result.hits[0].matched_fields["medication_names"] == ["metformin"]
    assert result.hits[0].matched_fields["diagnosis_terms"] == ["Type 2 diabetes mellitus"]


def test_bm25_retrieves_exact_icd_and_physician_terms(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        expected = seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000201"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000201"),
            patient_ref="PATIENT_REF_0201",
            content="Dr. Asha Raman documented hypertension with ICD-10 I10.",
            diagnosis="Hypertension",
            medications=["lisinopril 10 mg daily"],
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
        )
        seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000202"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000202"),
            patient_ref="PATIENT_REF_0202",
            content="Dr. Priya Nair documented diabetes with ICD-10 E11.9.",
            diagnosis="Type 2 diabetes mellitus",
            medications=["metformin 500 mg BID"],
            icd_codes=["E11.9"],
            physician="Dr. Priya Nair",
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="I10 records by Dr. Asha Raman",
        )

    assert result.hits[0].chunk_id == expected.id
    assert result.hits[0].matched_fields["icd_codes"] == ["I10"]
    assert result.hits[0].matched_fields["physician_names"] == ["Dr. Asha Raman"]


def test_bm25_resolves_mrn_tokens_for_authorized_identifier_search(
    session_local: sessionmaker[Session],
) -> None:
    patient_ref = "PATIENT_REF_0301"
    token = token_for_phi(entity_type="MRN", value="MRN-100100", patient_ref=patient_ref)

    with session_local() as db:
        db.add(
            PhiMapping(
                patient_ref=patient_ref,
                token=token,
                entity_type="MRN",
                encrypted_value=encrypt_phi_value("MRN-100100", token=token),
                encryption_key_id="local-development-key",
            )
        )
        expected = seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000301"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000301"),
            patient_ref=patient_ref,
            content=f"Patient reference {patient_ref}. MRN token {token}. Diagnosis: Hypertension.",
            diagnosis="Hypertension",
            medications=["lisinopril 10 mg daily"],
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.DOCTOR),
            query="MRN-100100",
        )

    assert result.query_terms.mrn_tokens == [token]
    assert result.query_terms.mrn_patient_refs == [patient_ref]
    assert result.hits[0].chunk_id == expected.id
    assert result.hits[0].matched_fields["mrn_tokens"] == [token]


def test_bm25_does_not_resolve_mrn_for_deidentified_role(
    session_local: sessionmaker[Session],
) -> None:
    patient_ref = "PATIENT_REF_0401"
    token = token_for_phi(entity_type="MRN", value="MRN-200200", patient_ref=patient_ref)

    with session_local() as db:
        db.add(
            PhiMapping(
                patient_ref=patient_ref,
                token=token,
                entity_type="MRN",
                encrypted_value=encrypt_phi_value("MRN-200200", token=token),
                encryption_key_id="local-development-key",
            )
        )
        seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000401"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000401"),
            patient_ref=patient_ref,
            content=f"MRN token {token}. Diagnosis: Hypertension.",
            diagnosis="Hypertension",
            medications=["lisinopril 10 mg daily"],
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
            sensitivity_level=SensitivityLevel.LOW,
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.RESEARCHER),
            query="MRN-200200",
        )

    assert result.query_terms.mrn_tokens == []
    assert result.hits == []


def test_bm25_applies_authorization_filters_before_scoring(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000501"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000501"),
            patient_ref="PATIENT_REF_0501",
            content="Restricted diabetes note with metformin.",
            diagnosis="Type 2 diabetes mellitus",
            medications=["metformin 500 mg BID"],
            icd_codes=["E11.9"],
            physician="Dr. Priya Nair",
            sensitivity_level=SensitivityLevel.RESTRICTED,
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.RESEARCHER),
            query="diabetes metformin",
        )

    assert result.authorization.allowed_sensitivity_levels == ["low", "medium"]
    assert result.candidate_chunk_count == 0
    assert result.hits == []


def test_bm25_scores_uploaded_prescription_facility_terms(
    session_local: sessionmaker[Session],
) -> None:
    with session_local() as db:
        expected = seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000601"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000601"),
            patient_ref="PATIENT_REF_0601",
            content=(
                "DD FORM 1289 DOD PRESCRIPTION. Medical Facility: "
                "U.S.S. Never forgotten (DD 178). Sig: 5 ml t.i.d a.c."
            ),
            diagnosis="",
            medications=["Belladonna 15 ml"],
            icd_codes=[],
            physician="Jack R Frost",
            hospital="U.S.S. Never forgotten (DD 178)",
            document_type=DocumentType.OTHER,
        )
        seed_chunk(
            db,
            document_id=UUID("41000000-0000-4000-8000-000000000602"),
            chunk_id=UUID("51000000-0000-4000-8000-000000000602"),
            patient_ref="PATIENT_REF_0602",
            content="Routine prescription refill from Northlake Medical Center.",
            diagnosis="Hypertension",
            medications=["lisinopril 10 mg daily"],
            icd_codes=["I10"],
            physician="Dr. Asha Raman",
            document_type=DocumentType.PRESCRIPTION,
        )

        result = bm25_search(
            db,
            user=user_by_role(db, RoleName.RECORDS_STAFF),
            query="prescriptions from U.S.S. Neverforgotten",
        )

    assert result.hits[0].chunk_id == expected.id
    assert "uss" in result.hits[0].matched_fields["content_terms"]
    assert "neverforgotten" in result.hits[0].matched_fields["content_terms"]
    assert result.hits[0].exact_match_score > result.hits[1].exact_match_score
