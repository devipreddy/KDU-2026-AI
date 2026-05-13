from collections.abc import Sequence
from pathlib import Path

from app.services.retrieval_quality_benchmarks import (
    BenchmarkQuery,
    BenchmarkThresholds,
    RetrievalBenchmarkHit,
    build_label_baseline_retriever,
    load_ground_truth_queries,
    load_synthetic_records,
    masking_benchmark_metrics,
    ocr_benchmark_metrics,
    run_retrieval_quality_benchmark,
)


class StepTimer:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def benchmark_query(
    query_id: str,
    *,
    expected_record_ids: list[str],
    expected_patient_refs: list[str],
) -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id=query_id,
        query=f"query {query_id}",
        positive_label=f"label_{query_id}",
        expected_record_ids=expected_record_ids,
        expected_patient_refs=expected_patient_refs,
    )


def hit(
    record_id: str,
    patient_ref: str,
    *,
    rank: int,
    rendered_text: str = "[DEID_PATIENT] has cardiac follow-up.",
    masking_mode: str = "de_identified",
    expected_phi_values: list[str] | None = None,
) -> RetrievalBenchmarkHit:
    return RetrievalBenchmarkHit(
        record_id=record_id,
        patient_ref=patient_ref,
        rank=rank,
        rendered_text=rendered_text,
        masking_mode=masking_mode,
        expected_phi_values=expected_phi_values or ["Jane Smith", "MRN-100001", "1972-04-08"],
    )


def test_retrieval_quality_benchmark_measures_top3_wrong_patient_latency_and_masking() -> None:
    queries = [
        benchmark_query(
            "q1",
            expected_record_ids=["REC-001"],
            expected_patient_refs=["PATIENT_REF_001"],
        ),
        benchmark_query(
            "q2",
            expected_record_ids=["REC-010"],
            expected_patient_refs=["PATIENT_REF_010"],
        ),
    ]

    def retriever(query: BenchmarkQuery, *, limit: int) -> Sequence[RetrievalBenchmarkHit]:
        assert limit == 3
        if query.query_id == "q1":
            return [
                hit("REC-404", "PATIENT_REF_999", rank=1),
                hit("REC-001", "PATIENT_REF_001", rank=2),
                hit("REC-002", "PATIENT_REF_001", rank=3),
            ]
        return [
            hit("REC-011", "PATIENT_REF_010", rank=1),
            hit("REC-012", "PATIENT_REF_010", rank=2),
            hit("REC-013", "PATIENT_REF_010", rank=3),
        ]

    report = run_retrieval_quality_benchmark(
        queries=queries,
        retriever=retriever,
        records=[
            {"ocr_required": True, "ocr_confidence": 0.91},
            {"ocr_required": True, "ocr_confidence": 0.60},
            {"ocr_required": False, "ocr_confidence": 0.99},
        ],
        timer=StepTimer([0.000, 0.012, 0.020, 0.052]),
        thresholds=BenchmarkThresholds(
            top3_accuracy_min=0.5,
            wrong_patient_retrieval_rate_max=0.2,
            p95_latency_ms_max=60,
            ocr_success_rate_min=0.5,
            masking_correctness_min=1.0,
        ),
    )

    assert report.top3_accuracy == 0.5
    assert report.wrong_patient_retrieval_rate == 0.166667
    assert report.average_latency_ms == 22.0
    assert report.p50_latency_ms == 12.0
    assert report.p95_latency_ms == 32.0
    assert report.ocr.evaluated_document_count == 2
    assert report.ocr.successful_document_count == 1
    assert report.ocr.success_rate == 0.5
    assert report.masking.evaluated_sample_count == 6
    assert report.masking.correctness_rate == 1.0
    assert report.passes_thresholds is True
    assert report.per_query[0].top3_hit is True
    assert report.per_query[0].wrong_patient_hits == 1
    assert report.per_query[1].top3_hit is False


def test_masking_benchmark_flags_leaked_phi_values_and_direct_patterns() -> None:
    runs = [
        type(
            "Run",
            (),
            {
                "hits": [
                    hit("REC-001", "PATIENT_REF_001", rank=1),
                    hit(
                        "REC-000",
                        "PATIENT_REF_000",
                        rank=0,
                        rendered_text="Visit Date: 2025-02-14 with de-identified text.",
                        expected_phi_values=[],
                    ),
                    hit(
                        "REC-002",
                        "PATIENT_REF_002",
                        rank=2,
                        rendered_text="Jane Smith MRN-100001 DOB 1972-04-08",
                    ),
                    hit(
                        "REC-003",
                        "PATIENT_REF_003",
                        rank=3,
                        rendered_text="Contact jane@example.com at 555-123-4567",
                        expected_phi_values=[],
                    ),
                    hit(
                        "REC-004",
                        "PATIENT_REF_004",
                        rank=4,
                        rendered_text=None,
                        masking_mode="metadata_only",
                    ),
                    hit(
                        "REC-005",
                        "PATIENT_REF_005",
                        rank=5,
                        rendered_text="Jane Smith appears for treating clinician",
                        masking_mode="full_phi",
                    ),
                ]
            },
        )()
    ]

    metrics = masking_benchmark_metrics(runs)

    assert metrics.evaluated_sample_count == 5
    assert metrics.passed_sample_count == 3
    assert metrics.correctness_rate == 0.6
    assert metrics.leaked_sample_ids == ["REC-002", "REC-003"]


def test_ocr_benchmark_prefers_explicit_success_status_then_confidence() -> None:
    metrics = ocr_benchmark_metrics(
        [
            {"ocr_required": True, "ocr_success": True, "ocr_confidence": 0.2},
            {"ocr_required": True, "ocr_status": "succeeded", "ocr_confidence": 0.2},
            {"ocr_required": True, "ocr_status": "failed", "ocr_confidence": 0.99},
            {"ocr_required": True, "ocr_confidence": 0.81},
            {"ocr_required": True, "ocr_confidence": 0.79},
            {"ocr_required": False, "ocr_confidence": 0.1},
        ],
        threshold=0.8,
    )

    assert metrics.evaluated_document_count == 5
    assert metrics.successful_document_count == 3
    assert metrics.success_rate == 0.6


def test_synthetic_dataset_loading_and_label_baseline_retriever() -> None:
    records = load_synthetic_records(Path("../data/synthetic/records.jsonl"))
    queries = load_ground_truth_queries(Path("../data/synthetic/ground_truth_queries.json"))
    retriever = build_label_baseline_retriever(records)

    first_query = queries[0]
    hits = list(retriever(first_query, limit=3))

    assert len(records) == 1000
    assert len(queries) >= 8
    assert len(hits) == 3
    assert hits[0].record_id in first_query.expected_record_set
    assert hits[0].patient_ref in first_query.expected_patient_set
    assert "MRN-" not in (hits[0].rendered_text or "")
    assert "Date of Birth:" in (hits[0].rendered_text or "")
    assert "[DEID_DOB]" in (hits[0].rendered_text or "")
