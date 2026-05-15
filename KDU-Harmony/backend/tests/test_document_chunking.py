from app.services.clinical_metadata import DocumentSection, extract_clinical_metadata
from app.services.document_chunking import plan_section_chunks


def test_plan_section_chunks_preserves_sections_and_child_overlap() -> None:
    text = (
        "Diagnosis:\n\nHypertension with chest pressure.\n\n"
        "Treatment Plan:\n\n"
        "Continue metoprolol today and repeat ECG in two weeks with cardiology follow up."
    )
    clinical_metadata = extract_clinical_metadata(text)

    planned_sections = plan_section_chunks(
        text,
        clinical_metadata.document_sections,
        child_token_target=6,
        child_token_overlap=2,
    )

    assert [section.section for section in planned_sections] == ["Diagnosis", "Treatment Plan"]
    assert (
        planned_sections[0].children[0].content == "Diagnosis:\n\nHypertension with chest pressure."
    )
    assert len(planned_sections[1].children) >= 2
    assert "repeat ECG" in planned_sections[1].children[1].content
    assert planned_sections[1].children[0].end_offset > planned_sections[1].children[1].start_offset


def test_plan_section_chunks_falls_back_to_document_section() -> None:
    text = "Unstructured note with hypertension and metoprolol."

    planned_sections = plan_section_chunks(
        text,
        [
            DocumentSection(
                section="",
                order=0,
                start_offset=0,
                end_offset=0,
                char_count=0,
            )
        ],
    )

    assert len(planned_sections) == 1
    assert planned_sections[0].section == "Document"
    assert planned_sections[0].children[0].content == text
