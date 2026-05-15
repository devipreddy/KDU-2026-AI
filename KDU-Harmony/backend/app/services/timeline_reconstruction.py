from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.user import User
from app.services.cross_encoder_reranking import DEFAULT_RERANK_TOP_N, CrossEncoderReranker
from app.services.embedding_pipeline import EmbeddingEncoder
from app.services.hybrid_retrieval import DEFAULT_CANDIDATE_LIMIT, DEFAULT_RRF_K
from app.services.query_understanding import QueryUnderstandingResult

TIMELINE_RECONSTRUCTION_VERSION = "patient_timeline_reconstruction_v1"
UNKNOWN_VALUE = "unknown"


@dataclass(frozen=True)
class TimelineSource:
    final_rank: int
    document_id: str
    external_id: str
    source_document: str
    visit_id: str | None
    section: str | None
    page_number: int | None
    chunk_id: str
    confidence_score: float
    confidence_level: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "document_id": self.document_id,
            "external_id": self.external_id,
            "source_document": self.source_document,
            "visit_id": self.visit_id,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class PatientTimelineGroup:
    patient_ref: str | None
    visit_date: str | None
    hospital: str | None
    diagnosis: str | None
    document_type: str | None
    visit_ids: list[str]
    document_ids: list[str]
    chunk_ids: list[str]
    source_documents: list[str]
    sections: list[str]
    icd_codes: list[str]
    highest_confidence: float
    confidence_level: str
    first_rank: int
    result_count: int
    sources: list[TimelineSource]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "patient_ref": self.patient_ref,
            "visit_date": self.visit_date,
            "hospital": self.hospital,
            "diagnosis": self.diagnosis,
            "document_type": self.document_type,
            "visit_ids": self.visit_ids,
            "document_ids": self.document_ids,
            "chunk_ids": self.chunk_ids,
            "source_documents": self.source_documents,
            "sections": self.sections,
            "icd_codes": self.icd_codes,
            "highest_confidence": self.highest_confidence,
            "confidence_level": self.confidence_level,
            "first_rank": self.first_rank,
            "result_count": self.result_count,
            "sources": [source.to_metadata() for source in self.sources],
        }


@dataclass(frozen=True)
class TimelineGroupKey:
    patient_ref: str | None
    visit_date: str | None
    hospital: str | None
    diagnosis: str | None
    document_type: str | None


def reconstruct_patient_timeline(search_result: Any) -> list[PatientTimelineGroup]:
    buckets: dict[TimelineGroupKey, list[Any]] = {}
    for hit in getattr(search_result, "hits", []):
        key = timeline_key_for_hit(hit)
        buckets.setdefault(key, []).append(hit)

    groups = [
        timeline_group_from_hits(key=key, hits=group_hits)
        for key, group_hits in buckets.items()
        if group_hits
    ]
    return sorted(groups, key=timeline_sort_key)


def timeline_key_for_hit(hit: Any) -> TimelineGroupKey:
    citation = hit.citation
    return TimelineGroupKey(
        patient_ref=patient_ref_for_hit(hit),
        visit_date=nullable_string(getattr(citation, "visit_date", None)),
        hospital=nullable_string(getattr(citation, "hospital", None)),
        diagnosis=nullable_string(getattr(citation, "diagnosis", None)),
        document_type=nullable_string(getattr(citation, "document_type", None)),
    )


def timeline_group_from_hits(
    *,
    key: TimelineGroupKey,
    hits: list[Any],
) -> PatientTimelineGroup:
    sorted_hits = sorted(hits, key=lambda hit: getattr(hit, "final_rank", 0))
    sources = [timeline_source_for_hit(hit) for hit in sorted_hits]
    highest_source = max(sources, key=lambda source: source.confidence_score)

    return PatientTimelineGroup(
        patient_ref=key.patient_ref,
        visit_date=key.visit_date,
        hospital=key.hospital,
        diagnosis=key.diagnosis,
        document_type=key.document_type,
        visit_ids=ordered_unique(source.visit_id for source in sources if source.visit_id),
        document_ids=ordered_unique(source.document_id for source in sources),
        chunk_ids=ordered_unique(source.chunk_id for source in sources),
        source_documents=ordered_unique(source.source_document for source in sources),
        sections=ordered_unique(source.section for source in sources if source.section),
        icd_codes=ordered_unique(
            code
            for hit in sorted_hits
            for code in list_values(getattr(hit.citation, "icd_codes", []))
        ),
        highest_confidence=highest_source.confidence_score,
        confidence_level=highest_source.confidence_level,
        first_rank=min(source.final_rank for source in sources),
        result_count=len(sorted_hits),
        sources=sources,
    )


def timeline_source_for_hit(hit: Any) -> TimelineSource:
    citation = hit.citation
    confidence = hit.confidence
    return TimelineSource(
        final_rank=int(getattr(hit, "final_rank", 0)),
        document_id=str(getattr(citation, "document_id", "")),
        external_id=string_or_empty(getattr(citation, "external_id", "")),
        source_document=string_or_empty(getattr(citation, "source_document", "")),
        visit_id=nullable_string(getattr(citation, "visit_id", None)),
        section=nullable_string(getattr(citation, "section", None)),
        page_number=getattr(citation, "page_number", None),
        chunk_id=str(getattr(hit.matched_chunk, "chunk_id", "")),
        confidence_score=float(getattr(confidence, "score", 0.0)),
        confidence_level=string_or_empty(getattr(confidence, "level", UNKNOWN_VALUE)),
    )


def patient_ref_for_hit(hit: Any) -> str | None:
    if hasattr(hit, "patient_display_ref"):
        return nullable_string(getattr(hit, "patient_display_ref", None))
    return nullable_string(getattr(hit.citation, "patient_ref", None))


def timeline_sort_key(group: PatientTimelineGroup) -> tuple[str, str, str, str, str, int]:
    return (
        group.patient_ref or UNKNOWN_VALUE,
        group.visit_date or "9999-12-31",
        group.hospital or UNKNOWN_VALUE,
        group.diagnosis or UNKNOWN_VALUE,
        group.document_type or UNKNOWN_VALUE,
        group.first_rank,
    )


def ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def string_or_empty(value: Any) -> str:
    return nullable_string(value) or ""


def list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def timeline_search(
    db,
    *,
    user: User,
    query: str | QueryUnderstandingResult,
    limit: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    rrf_k: int = DEFAULT_RRF_K,
    authorized_patient_refs: list[str] | None = None,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[PatientTimelineGroup]:
    from app.services.context_expansion import contextual_search

    context_result = contextual_search(
        db,
        user=user,
        query=query,
        limit=limit,
        candidate_limit=candidate_limit,
        rerank_top_n=rerank_top_n,
        rrf_k=rrf_k,
        authorized_patient_refs=authorized_patient_refs,
        encoder=encoder,
        collection=collection,
        collection_name=collection_name,
        reranker=reranker,
    )
    return reconstruct_patient_timeline(context_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run search and reconstruct patient timelines.")
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--collection", default=settings.chroma_collection)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        user = db.scalar(
            select(User).options(selectinload(User.roles)).where(User.email == args.email)
        )
        if user is None:
            raise SystemExit(f"User not found: {args.email}")
        timeline = timeline_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            rerank_top_n=args.rerank_top_n,
            rrf_k=args.rrf_k,
            collection_name=args.collection,
        )
    print(
        json.dumps(
            {
                "timeline_reconstructor": TIMELINE_RECONSTRUCTION_VERSION,
                "group_count": len(timeline),
                "groups": [group.to_metadata() for group in timeline],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
