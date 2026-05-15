from app.services.clinical_metadata import extract_clinical_metadata


def test_extract_clinical_metadata_from_structured_note() -> None:
    text = """
Visit Date: 2025-02-14
Hospital: Harmony General Hospital
Physician: Dr. Asha Raman

Chief Complaint:

Patient reports chest pressure.

Diagnosis: Hypertension

ICD-10: I10

Medications: metoprolol 25 mg BID

Assessment:

Findings are consistent with hypertension and type 2 diabetes.

Plan:

Continue metformin and repeat ECG.
""".strip()

    result = extract_clinical_metadata(text)
    metadata = result.to_metadata()

    assert result.diagnoses == ["Hypertension", "Type 2 diabetes mellitus"]
    assert "metoprolol 25 mg BID" in result.medications
    assert "metformin" in result.medications
    assert result.symptoms == ["chest pressure"]
    assert result.icd_codes == ["I10"]
    assert result.physicians == ["Dr. Asha Raman"]
    assert result.hospitals == ["Harmony General Hospital"]
    assert metadata["dates"] == [{"label": "visit_date", "value": "2025-02-14"}]
    assert [section.section for section in result.document_sections] == [
        "Chief Complaint",
        "Diagnosis",
        "ICD-10",
        "Medications",
        "Assessment",
        "Plan",
    ]
    assert metadata["entity_counts"]["document_sections"] == 6


def test_extract_clinical_metadata_handles_ocr_like_unstructured_text() -> None:
    result = extract_clinical_metadata("Diagnosis Hypertension Beta blockers Visit Date 2025-03-03")

    assert result.diagnoses == ["Hypertension"]
    assert result.medications == ["Beta blockers"]
    assert result.to_metadata()["dates"] == [{"label": "date", "value": "2025-03-03"}]


def test_extract_clinical_metadata_recognizes_malaria_terms() -> None:
    text = """
Visit Date: 2025-06-18
Diagnosis: Malaria
ICD-10: B54
Medication: artemether lumefantrine BID
Chief Complaint: Patient reports cyclic fever with chills.
""".strip()

    result = extract_clinical_metadata(text)

    assert result.diagnoses == ["Malaria"]
    assert "artemether lumefantrine BID" in result.medications
    assert "cyclic fever with chills" in result.symptoms
    assert result.icd_codes == ["B54"]
