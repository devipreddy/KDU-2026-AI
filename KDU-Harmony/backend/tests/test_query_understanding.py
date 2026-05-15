from app.services.bm25_retrieval import tokenize_lexical_text
from app.services.query_understanding import understand_query


def test_understand_query_extracts_cardiac_q1_temporal_filter() -> None:
    result = understand_query("patients with cardiac issues treated in Q1 2025")

    assert result.temporal_filters[0].operator == "between"
    assert result.temporal_filters[0].start_date == "2025-01-01"
    assert result.temporal_filters[0].end_date == "2025-03-31"
    assert result.temporal_filters[0].granularity == "quarter"
    assert result.diagnosis_concepts[0].concept == "cardiac"
    assert result.diagnosis_concepts[0].concept_type == "category"
    assert "Hypertension" in result.diagnosis_concepts[0].diagnoses
    assert result.metadata_filters["diagnosis_category"] == ["cardiac"]
    assert result.metadata_filters["visit_date"][0]["matched_text"] == "Q1 2025"


def test_understand_query_extracts_patient_provider_facility_and_document_type() -> None:
    result = understand_query(
        "Show patient ref 42 discharge summaries from Harmony General Hospital "
        "by Dr. Asha Raman after Jan 1, 2025"
    )

    assert result.patient_refs == ["PATIENT_REF_0042"]
    assert result.hospitals == ["Harmony General Hospital"]
    assert result.physicians == ["Dr. Asha Raman"]
    assert result.document_types == ["discharge_summary"]
    assert result.temporal_filters[0].operator == "gte"
    assert result.temporal_filters[0].start_date == "2025-01-01"
    assert result.temporal_filters[0].end_date is None
    assert result.metadata_filters["patient_ref"] == ["PATIENT_REF_0042"]
    assert result.metadata_filters["document_type"] == ["discharge_summary"]


def test_understand_query_extracts_diagnosis_aliases_and_year_filters() -> None:
    result = understand_query("scanned documents with chronic kidney disease during 2025")

    assert result.document_types == ["scanned_pdf"]
    assert result.temporal_filters[0].granularity == "year"
    assert result.temporal_filters[0].start_date == "2025-01-01"
    assert result.temporal_filters[0].end_date == "2025-12-31"
    assert result.diagnosis_concepts[0].concept == "Chronic kidney disease stage 3"
    assert result.metadata_filters["diagnosis"] == ["Chronic kidney disease stage 3"]
    assert result.metadata_filters["diagnosis_category"] == ["renal"]


def test_understand_query_extracts_between_dates_icd_and_lab_document_type() -> None:
    result = understand_query(
        "lab reports for I10 at Mercy West Health between 02/01/2025 and 03/15/2025"
    )

    assert result.document_types == ["lab_report"]
    assert result.icd_codes == ["I10"]
    assert result.hospitals == ["Mercy West Health"]
    assert result.temporal_filters[0].operator == "between"
    assert result.temporal_filters[0].start_date == "2025-02-01"
    assert result.temporal_filters[0].end_date == "2025-03-15"
    assert result.metadata_filters["icd_codes"] == ["I10"]


def test_understand_query_extracts_malaria_diagnosis() -> None:
    result = understand_query("patients with malaria")

    assert result.diagnosis_concepts[0].concept == "Malaria"
    assert result.diagnosis_concepts[0].concept_type == "diagnosis"
    assert result.metadata_filters["diagnosis"] == ["Malaria"]


def test_understand_query_expands_prescription_forms_across_source_types() -> None:
    result = understand_query("prescriptions from U.S.S. Neverforgotten")

    assert result.document_types == [
        "prescription",
        "handwritten_note",
        "scanned_pdf",
        "typed_pdf",
        "other",
    ]
    assert result.hospitals == []
    assert result.metadata_filters["document_type"] == result.document_types
    assert "hospital" not in result.metadata_filters


def test_understand_query_maps_common_clinical_language_to_diagnoses() -> None:
    result = understand_query("patients with high blood pressure and irregular heartbeat")

    diagnoses = [concept.concept for concept in result.diagnosis_concepts]
    assert "Hypertension" in diagnoses
    assert "Atrial fibrillation" in diagnoses


def test_lexical_tokens_tolerate_ocr_spacing_and_punctuation_variants() -> None:
    document_tokens = tokenize_lexical_text("U.S.S. Never forgotten DOD PRESCRIPTION")
    query_tokens = tokenize_lexical_text("USS Neverforgotten prescriptions")

    assert "uss" in document_tokens
    assert "neverforgotten" in document_tokens
    assert "prescription" in query_tokens


def test_understand_query_serializes_to_metadata() -> None:
    result = understand_query("respiratory visits during 2025")
    metadata = result.to_metadata()

    assert metadata["parser"] == "rule_based_query_understanding_v1"
    assert metadata["diagnosis_concepts"][0]["concept"] == "respiratory"
    assert metadata["metadata_filters"]["diagnosis_category"] == ["respiratory"]
