from typing import Any, cast
from uuid import UUID

from app.services.context_expansion import (
    ChunkContext,
    ContextualSearchHit,
    ContextualSearchResult,
    RetrievalConfidence,
    SourceCitation,
)
from app.services.timeline_reconstruction import reconstruct_patient_timeline


class DummyAuthorization:
    def to_metadata(self) -> dict[str, Any]:
        return {"denied": False}


class DummyRerankResult:
    reranker_model = "unit-test-reranker"


def build_hit(
    *,
    rank: int,
    patient_ref: str,
    visit_date: str,
    hospital: str,
    diagnosis: str,
    document_type: str,
    document_id: UUID,
    chunk_id: UUID,
    section: str,
    confidence_score: float,
) -> ContextualSearchHit:
    return ContextualSearchHit(
        final_rank=rank,
        matched_chunk=ChunkContext(
            chunk_id=chunk_id,
            section=section,
            text=f"{diagnosis} timeline source text.",
            page_number=rank,
            start_offset=0,
            end_offset=40,
            token_count=5,
            chunk_type="child",
        ),
        parent_context=None,
        citation=SourceCitation(
            document_id=document_id,
            external_id=f"DOC-{str(document_id)[-4:]}",
            source_document=f"{str(document_id)[-4:]}.pdf",
            source_uri=f"local-encrypted://{document_id}",
            document_type=document_type,
            page_number=rank,
            section=section,
            hospital=hospital,
            physician="Dr. Asha Raman",
            visit_id=f"VISIT-{visit_date}",
            checksum_sha256="a" * 64,
            citation_label=f"DOC-{str(document_id)[-4:]} | p. {rank} | {section}",
            patient_ref=patient_ref,
            diagnosis=diagnosis,
            icd_codes=["I48.91" if "fibrillation" in diagnosis.lower() else "I20.9"],
            visit_date=visit_date,
        ),
        confidence=RetrievalConfidence(
            score=confidence_score,
            level="high" if confidence_score >= 0.85 else "medium",
            reranker_score=confidence_score,
            hybrid_score=0.03,
            ocr_confidence=0.94,
            source_count=2,
        ),
        retrieval={"sources": ["bm25", "dense"], "final_rank": rank},
    )


def build_context_result(hits: list[ContextualSearchHit]) -> ContextualSearchResult:
    return ContextualSearchResult(
        query="patients with cardiac issues treated in Q1 2025",
        authorization=cast(Any, DummyAuthorization()),
        hits=hits,
        rerank_result=cast(Any, DummyRerankResult()),
    )


def test_reconstruct_patient_timeline_groups_by_patient_visit_and_document_metadata() -> None:
    result = build_context_result(
        [
            build_hit(
                rank=2,
                patient_ref="PATIENT_REF_42",
                visit_date="2025-02-14",
                hospital="Harmony General Hospital",
                diagnosis="Atrial fibrillation",
                document_type="clinical_note",
                document_id=UUID("41000000-0000-4000-8000-000000002701"),
                chunk_id=UUID("51000000-0000-4000-8000-000000002701"),
                section="Assessment",
                confidence_score=0.87,
            ),
            build_hit(
                rank=1,
                patient_ref="PATIENT_REF_42",
                visit_date="2025-02-14",
                hospital="Harmony General Hospital",
                diagnosis="Atrial fibrillation",
                document_type="clinical_note",
                document_id=UUID("41000000-0000-4000-8000-000000002701"),
                chunk_id=UUID("51000000-0000-4000-8000-000000002702"),
                section="Treatment Plan",
                confidence_score=0.94,
            ),
            build_hit(
                rank=3,
                patient_ref="PATIENT_REF_42",
                visit_date="2025-03-03",
                hospital="North Valley Medical Center",
                diagnosis="Stable angina",
                document_type="discharge_summary",
                document_id=UUID("41000000-0000-4000-8000-000000002703"),
                chunk_id=UUID("51000000-0000-4000-8000-000000002703"),
                section="Discharge Summary",
                confidence_score=0.89,
            ),
            build_hit(
                rank=4,
                patient_ref="PATIENT_REF_84",
                visit_date="2025-01-17",
                hospital="Lakeview Regional Hospital",
                diagnosis="Chronic kidney disease stage 3",
                document_type="clinical_note",
                document_id=UUID("41000000-0000-4000-8000-000000002704"),
                chunk_id=UUID("51000000-0000-4000-8000-000000002704"),
                section="Treatment Plan",
                confidence_score=0.72,
            ),
        ]
    )

    timeline = reconstruct_patient_timeline(result)

    assert len(timeline) == 3
    first_group = timeline[0]
    assert first_group.patient_ref == "PATIENT_REF_42"
    assert first_group.visit_date == "2025-02-14"
    assert first_group.hospital == "Harmony General Hospital"
    assert first_group.diagnosis == "Atrial fibrillation"
    assert first_group.document_type == "clinical_note"
    assert first_group.result_count == 2
    assert first_group.document_ids == ["41000000-0000-4000-8000-000000002701"]
    assert first_group.sections == ["Treatment Plan", "Assessment"]
    assert first_group.highest_confidence == 0.94
    assert first_group.confidence_level == "high"
    assert [source.final_rank for source in first_group.sources] == [1, 2]

    assert [(group.patient_ref, group.visit_date) for group in timeline] == [
        ("PATIENT_REF_42", "2025-02-14"),
        ("PATIENT_REF_42", "2025-03-03"),
        ("PATIENT_REF_84", "2025-01-17"),
    ]


def test_contextual_result_metadata_includes_timeline_groups() -> None:
    result = build_context_result(
        [
            build_hit(
                rank=1,
                patient_ref="PATIENT_REF_42",
                visit_date="2025-02-14",
                hospital="Harmony General Hospital",
                diagnosis="Atrial fibrillation",
                document_type="clinical_note",
                document_id=UUID("41000000-0000-4000-8000-000000002705"),
                chunk_id=UUID("51000000-0000-4000-8000-000000002705"),
                section="Assessment",
                confidence_score=0.91,
            )
        ]
    )

    metadata = result.to_metadata()

    assert metadata["timeline"][0]["patient_ref"] == "PATIENT_REF_42"
    assert metadata["timeline"][0]["visit_date"] == "2025-02-14"
    assert metadata["timeline"][0]["document_type"] == "clinical_note"
    assert metadata["timeline"][0]["sources"][0]["chunk_id"] == (
        "51000000-0000-4000-8000-000000002705"
    )
