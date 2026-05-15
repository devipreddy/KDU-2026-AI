from app.services.text_normalization import normalize_medical_text


def test_normalize_medical_text_cleans_artifacts_headings_and_misspellings() -> None:
    raw_text = """
    diagnosis - Hvpcrtension
    Plan::
    Continue Bcta blockers and monitor SOB.

    Medication
    Metforrnin 500 mg BID

    Assess-
    ment notes include Arrhythrnia follow up.  |  __
    """

    result = normalize_medical_text(raw_text)

    assert "Diagnosis: Hypertension" in result.text
    assert "Plan:" in result.text
    assert "Continue Beta blockers and monitor Shortness of breath." in result.text
    assert "Medications:" in result.text
    assert "Metformin 500 mg BID" in result.text
    assert "Assessment notes include Arrhythmia follow up." in result.text
    assert "|" not in result.text
    assert "__" not in result.text
    assert result.stats.heading_count == 3
    assert result.stats.artifact_replacements >= 2
    assert result.stats.medical_corrections["Hypertension"] == 1
    assert result.stats.medical_corrections["Beta blockers"] == 1


def test_normalize_medical_text_restores_paragraphs() -> None:
    raw_text = "Assessment:\nPatient reports\nchest pressure.\n\nPlan:\nRepeat ECG\nin two weeks."

    result = normalize_medical_text(raw_text)

    assert result.text == (
        "Assessment:\n\nPatient reports chest pressure.\n\nPlan:\n\nRepeat ECG in two weeks."
    )
    assert result.stats.paragraph_count == 4
