from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from app.core.observability import (
    EMAIL_PATTERN,
    MRN_PATTERN,
    PHONE_PATTERN,
    SSN_PATTERN,
    redact_phi_in_text,
)

RETRIEVAL_BENCHMARK_VERSION = "retrieval_quality_benchmark_v1"
DEFAULT_SYNTHETIC_DIR = Path(__file__).resolve().parents[3] / "data" / "synthetic"
DEFAULT_GROUND_TRUTH_PATH = DEFAULT_SYNTHETIC_DIR / "ground_truth_queries.json"
DEFAULT_RECORDS_PATH = DEFAULT_SYNTHETIC_DIR / "records.jsonl"
DEFAULT_TOP_K = 3
DEFAULT_OCR_SUCCESS_THRESHOLD = 0.80
DEFAULT_THRESHOLDS = {
    "top3_accuracy_min": 0.90,
    "wrong_patient_retrieval_rate_max": 0.0,
    "p95_latency_ms_max": 2_000.0,
    "ocr_success_rate_min": 0.95,
    "masking_correctness_min": 1.0,
}

DIRECT_DOB_PATTERN = re.compile(
    r"\b(?:DOB|date\s+of\s+birth)\s*[:#-]?\s*"
    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkThresholds:
    top3_accuracy_min: float = DEFAULT_THRESHOLDS["top3_accuracy_min"]
    wrong_patient_retrieval_rate_max: float = DEFAULT_THRESHOLDS["wrong_patient_retrieval_rate_max"]
    p95_latency_ms_max: float = DEFAULT_THRESHOLDS["p95_latency_ms_max"]
    ocr_success_rate_min: float = DEFAULT_THRESHOLDS["ocr_success_rate_min"]
    masking_correctness_min: float = DEFAULT_THRESHOLDS["masking_correctness_min"]

    def to_metadata(self) -> dict[str, float]:
        return {
            "top3_accuracy_min": self.top3_accuracy_min,
            "wrong_patient_retrieval_rate_max": self.wrong_patient_retrieval_rate_max,
            "p95_latency_ms_max": self.p95_latency_ms_max,
            "ocr_success_rate_min": self.ocr_success_rate_min,
            "masking_correctness_min": self.masking_correctness_min,
        }


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str
    positive_label: str
    expected_record_ids: list[str]
    expected_patient_refs: list[str]
    description: str | None = None

    @property
    def expected_record_set(self) -> set[str]:
        return set(self.expected_record_ids)

    @property
    def expected_patient_set(self) -> set[str]:
        return set(self.expected_patient_refs)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "positive_label": self.positive_label,
            "expected_record_count": len(self.expected_record_ids),
            "expected_patient_count": len(self.expected_patient_refs),
            "description": self.description,
        }


@dataclass(frozen=True)
class RetrievalBenchmarkHit:
    record_id: str
    patient_ref: str
    rank: int
    score: float | None = None
    document_id: str | None = None
    rendered_text: str | None = None
    masking_mode: str | None = None
    expected_phi_values: list[str] | None = None
    ocr_required: bool = False
    ocr_confidence: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "patient_ref": self.patient_ref,
            "rank": self.rank,
            "score": self.score,
            "document_id": self.document_id,
            "masking_mode": self.masking_mode,
            "ocr_required": self.ocr_required,
            "ocr_confidence": self.ocr_confidence,
        }


@dataclass(frozen=True)
class BenchmarkQueryRun:
    query: BenchmarkQuery
    hits: list[RetrievalBenchmarkHit]
    latency_ms: float

    @property
    def top_hits(self) -> list[RetrievalBenchmarkHit]:
        return sorted(self.hits, key=lambda hit: hit.rank)[:DEFAULT_TOP_K]


@dataclass(frozen=True)
class BenchmarkQueryMetrics:
    query_id: str
    query: str
    positive_label: str
    top3_hit: bool
    returned_count: int
    wrong_patient_hits: int
    evaluated_hits: int
    latency_ms: float
    top3_record_ids: list[str]
    expected_record_count: int
    expected_patient_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "positive_label": self.positive_label,
            "top3_hit": self.top3_hit,
            "returned_count": self.returned_count,
            "wrong_patient_hits": self.wrong_patient_hits,
            "evaluated_hits": self.evaluated_hits,
            "latency_ms": self.latency_ms,
            "top3_record_ids": self.top3_record_ids,
            "expected_record_count": self.expected_record_count,
            "expected_patient_count": self.expected_patient_count,
        }


@dataclass(frozen=True)
class OCRBenchmarkMetrics:
    evaluated_document_count: int
    successful_document_count: int
    success_rate: float
    threshold: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evaluated_document_count": self.evaluated_document_count,
            "successful_document_count": self.successful_document_count,
            "success_rate": self.success_rate,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class MaskingBenchmarkMetrics:
    evaluated_sample_count: int
    passed_sample_count: int
    correctness_rate: float
    leaked_sample_ids: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evaluated_sample_count": self.evaluated_sample_count,
            "passed_sample_count": self.passed_sample_count,
            "correctness_rate": self.correctness_rate,
            "leaked_sample_ids": self.leaked_sample_ids,
        }


@dataclass(frozen=True)
class RetrievalQualityBenchmarkReport:
    benchmark_version: str
    query_count: int
    top3_accuracy: float
    wrong_patient_retrieval_rate: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    ocr: OCRBenchmarkMetrics
    masking: MaskingBenchmarkMetrics
    per_query: list[BenchmarkQueryMetrics]
    thresholds: BenchmarkThresholds

    @property
    def passes_thresholds(self) -> bool:
        return (
            self.top3_accuracy >= self.thresholds.top3_accuracy_min
            and self.wrong_patient_retrieval_rate
            <= self.thresholds.wrong_patient_retrieval_rate_max
            and self.p95_latency_ms <= self.thresholds.p95_latency_ms_max
            and self.ocr.success_rate >= self.thresholds.ocr_success_rate_min
            and self.masking.correctness_rate >= self.thresholds.masking_correctness_min
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "query_count": self.query_count,
            "top3_accuracy": self.top3_accuracy,
            "wrong_patient_retrieval_rate": self.wrong_patient_retrieval_rate,
            "average_latency_ms": self.average_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "ocr": self.ocr.to_metadata(),
            "masking": self.masking.to_metadata(),
            "thresholds": self.thresholds.to_metadata(),
            "passes_thresholds": self.passes_thresholds,
            "per_query": [query.to_metadata() for query in self.per_query],
        }


class BenchmarkRetriever(Protocol):
    def __call__(self, query: BenchmarkQuery, *, limit: int) -> Sequence[RetrievalBenchmarkHit]:
        """Return ranked retrieval hits for a benchmark query."""


def run_retrieval_quality_benchmark(
    *,
    queries: Sequence[BenchmarkQuery],
    retriever: BenchmarkRetriever,
    records: Sequence[dict[str, Any]] = (),
    top_k: int = DEFAULT_TOP_K,
    ocr_success_threshold: float = DEFAULT_OCR_SUCCESS_THRESHOLD,
    thresholds: BenchmarkThresholds | None = None,
    timer: Callable[[], float] = perf_counter,
) -> RetrievalQualityBenchmarkReport:
    runs = [
        timed_query_run(query=query, retriever=retriever, limit=top_k, timer=timer)
        for query in queries
    ]
    per_query = [query_metrics(run, top_k=top_k) for run in runs]
    latency_values = [run.latency_ms for run in runs]
    evaluated_hits = sum(metric.evaluated_hits for metric in per_query)
    wrong_patient_hits = sum(metric.wrong_patient_hits for metric in per_query)

    return RetrievalQualityBenchmarkReport(
        benchmark_version=RETRIEVAL_BENCHMARK_VERSION,
        query_count=len(queries),
        top3_accuracy=round(
            safe_rate(sum(metric.top3_hit for metric in per_query), len(queries)), 6
        ),
        wrong_patient_retrieval_rate=round(
            safe_rate(wrong_patient_hits, evaluated_hits),
            6,
        ),
        average_latency_ms=round(sum(latency_values) / len(latency_values), 3)
        if latency_values
        else 0.0,
        p50_latency_ms=round(percentile(latency_values, 50), 3),
        p95_latency_ms=round(percentile(latency_values, 95), 3),
        ocr=ocr_benchmark_metrics(records, threshold=ocr_success_threshold),
        masking=masking_benchmark_metrics(runs),
        per_query=per_query,
        thresholds=thresholds or BenchmarkThresholds(),
    )


def timed_query_run(
    *,
    query: BenchmarkQuery,
    retriever: BenchmarkRetriever,
    limit: int,
    timer: Callable[[], float],
) -> BenchmarkQueryRun:
    started_at = timer()
    hits = list(retriever(query, limit=limit))
    latency_ms = (timer() - started_at) * 1000
    return BenchmarkQueryRun(
        query=query,
        hits=sorted(hits, key=lambda hit: hit.rank),
        latency_ms=round(latency_ms, 3),
    )


def query_metrics(run: BenchmarkQueryRun, *, top_k: int) -> BenchmarkQueryMetrics:
    top_hits = sorted(run.hits, key=lambda hit: hit.rank)[:top_k]
    expected_record_ids = run.query.expected_record_set
    expected_patient_refs = run.query.expected_patient_set
    wrong_patient_hits = sum(
        1
        for hit in top_hits
        if expected_patient_refs and hit.patient_ref not in expected_patient_refs
    )
    return BenchmarkQueryMetrics(
        query_id=run.query.query_id,
        query=run.query.query,
        positive_label=run.query.positive_label,
        top3_hit=any(hit.record_id in expected_record_ids for hit in top_hits),
        returned_count=len(run.hits),
        wrong_patient_hits=wrong_patient_hits,
        evaluated_hits=len(top_hits),
        latency_ms=run.latency_ms,
        top3_record_ids=[hit.record_id for hit in top_hits],
        expected_record_count=len(run.query.expected_record_ids),
        expected_patient_count=len(run.query.expected_patient_refs),
    )


def ocr_benchmark_metrics(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
) -> OCRBenchmarkMetrics:
    ocr_records = [record for record in records if bool(record.get("ocr_required"))]
    successful_count = sum(1 for record in ocr_records if ocr_record_succeeded(record, threshold))
    return OCRBenchmarkMetrics(
        evaluated_document_count=len(ocr_records),
        successful_document_count=successful_count,
        success_rate=round(safe_rate(successful_count, len(ocr_records), default=1.0), 6),
        threshold=threshold,
    )


def ocr_record_succeeded(record: dict[str, Any], threshold: float) -> bool:
    if "ocr_success" in record:
        return bool(record["ocr_success"])
    if "ocr_status" in record:
        return str(record["ocr_status"]).lower() in {"indexed", "processed", "succeeded"}
    confidence = record.get("ocr_confidence")
    if confidence is None:
        return False
    return float(confidence) >= threshold


def masking_benchmark_metrics(runs: Sequence[BenchmarkQueryRun]) -> MaskingBenchmarkMetrics:
    samples = [
        hit
        for run in runs
        for hit in run.hits
        if hit.masking_mode and hit.masking_mode != "full_phi"
    ]
    leaked_sample_ids = [hit.record_id for hit in samples if not masking_sample_is_correct(hit)]
    passed_count = len(samples) - len(leaked_sample_ids)
    return MaskingBenchmarkMetrics(
        evaluated_sample_count=len(samples),
        passed_sample_count=passed_count,
        correctness_rate=round(safe_rate(passed_count, len(samples), default=1.0), 6),
        leaked_sample_ids=leaked_sample_ids,
    )


def masking_sample_is_correct(hit: RetrievalBenchmarkHit) -> bool:
    text = hit.rendered_text
    if text is None:
        return True
    normalized_text = text.lower()
    for value in hit.expected_phi_values or []:
        if value and str(value).lower() in normalized_text:
            return False
    return not contains_direct_phi_pattern(text)


def contains_direct_phi_pattern(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (EMAIL_PATTERN, PHONE_PATTERN, SSN_PATTERN, MRN_PATTERN, DIRECT_DOB_PATTERN)
    )


def percentile(values: Sequence[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, math.ceil((percentile_value / 100) * len(sorted_values)) - 1)
    return sorted_values[index]


def safe_rate(numerator: int | float, denominator: int, *, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def load_ground_truth_queries(path: Path = DEFAULT_GROUND_TRUTH_PATH) -> list[BenchmarkQuery]:
    raw_queries = json.loads(path.read_text(encoding="utf-8"))
    return [
        BenchmarkQuery(
            query_id=item["query_id"],
            query=item["query"],
            positive_label=item["positive_label"],
            expected_record_ids=list(item["expected_record_ids"]),
            expected_patient_refs=list(item["expected_patient_refs"]),
            description=item.get("description"),
        )
        for item in raw_queries
    ]


def load_synthetic_records(path: Path = DEFAULT_RECORDS_PATH) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_label_baseline_retriever(
    records: Sequence[dict[str, Any]],
) -> BenchmarkRetriever:
    records_by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for label in record.get("ground_truth_labels", []):
            records_by_label.setdefault(str(label), []).append(record)

    def retrieve(query: BenchmarkQuery, *, limit: int) -> Sequence[RetrievalBenchmarkHit]:
        matching_records = records_by_label.get(query.positive_label, [])[:limit]
        return [
            hit_from_synthetic_record(record, rank=rank)
            for rank, record in enumerate(matching_records, start=1)
        ]

    return retrieve


def hit_from_synthetic_record(record: dict[str, Any], *, rank: int) -> RetrievalBenchmarkHit:
    return RetrievalBenchmarkHit(
        record_id=str(record["record_id"]),
        patient_ref=str(record["patient_ref"]),
        rank=rank,
        score=1 / rank,
        document_id=str(record.get("external_id") or record["record_id"]),
        rendered_text=deidentified_text_for_record(record),
        masking_mode="de_identified",
        expected_phi_values=synthetic_phi_values(record),
        ocr_required=bool(record.get("ocr_required")),
        ocr_confidence=float(record["ocr_confidence"]) if record.get("ocr_confidence") else None,
    )


def deidentified_text_for_record(record: dict[str, Any]) -> str:
    text = str(record.get("text", ""))
    for original, replacement in (
        (record.get("synthetic_patient_name"), "[DEID_PATIENT]"),
        (record.get("synthetic_mrn"), "[DEID_MRN]"),
        (record.get("synthetic_date_of_birth"), "[DEID_DOB]"),
        (record.get("patient_ref"), "[DEID_PATIENT_REF]"),
    ):
        if original:
            text = text.replace(str(original), replacement)
    return redact_phi_in_text(text)


def synthetic_phi_values(record: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for value in (
            record.get("synthetic_patient_name"),
            record.get("synthetic_mrn"),
            record.get("synthetic_date_of_birth"),
            record.get("patient_ref"),
        )
        if value
    ]


def run_synthetic_label_benchmark(
    *,
    records_path: Path = DEFAULT_RECORDS_PATH,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
    thresholds: BenchmarkThresholds | None = None,
) -> RetrievalQualityBenchmarkReport:
    records = load_synthetic_records(records_path)
    queries = load_ground_truth_queries(ground_truth_path)
    return run_retrieval_quality_benchmark(
        queries=queries,
        retriever=build_label_baseline_retriever(records),
        records=records,
        thresholds=thresholds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval quality benchmarks against synthetic ground truth."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--top3-min", type=float, default=DEFAULT_THRESHOLDS["top3_accuracy_min"])
    parser.add_argument(
        "--wrong-patient-max",
        type=float,
        default=DEFAULT_THRESHOLDS["wrong_patient_retrieval_rate_max"],
    )
    parser.add_argument(
        "--p95-latency-max-ms",
        type=float,
        default=DEFAULT_THRESHOLDS["p95_latency_ms_max"],
    )
    parser.add_argument(
        "--ocr-success-min",
        type=float,
        default=DEFAULT_THRESHOLDS["ocr_success_rate_min"],
    )
    parser.add_argument(
        "--masking-min",
        type=float,
        default=DEFAULT_THRESHOLDS["masking_correctness_min"],
    )
    parser.add_argument("--fail-on-thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = BenchmarkThresholds(
        top3_accuracy_min=args.top3_min,
        wrong_patient_retrieval_rate_max=args.wrong_patient_max,
        p95_latency_ms_max=args.p95_latency_max_ms,
        ocr_success_rate_min=args.ocr_success_min,
        masking_correctness_min=args.masking_min,
    )
    report = run_synthetic_label_benchmark(
        records_path=args.records,
        ground_truth_path=args.ground_truth,
        thresholds=thresholds,
    )
    print(json.dumps(report.to_metadata(), indent=2, sort_keys=True))
    if args.fail_on_thresholds and not report.passes_thresholds:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
