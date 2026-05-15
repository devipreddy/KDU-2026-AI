from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.services.cross_encoder_reranking import (
    DEFAULT_RERANK_TOP_N,
    CrossEncoderReranker,
    CrossEncoderRerankResult,
    RerankedSearchHit,
    reranked_hybrid_search,
)
from app.services.embedding_pipeline import EmbeddingEncoder
from app.services.hybrid_retrieval import DEFAULT_CANDIDATE_LIMIT, DEFAULT_RRF_K
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import AuthorizedMetadataFilter

CONTEXT_EXPANSION_VERSION = "parent_context_expansion_v1"


@dataclass(frozen=True)
class ChunkContext:
    chunk_id: uuid.UUID
    section: str | None
    text: str
    page_number: int | None
    start_offset: int | None
    end_offset: int | None
    token_count: int | None
    chunk_type: str | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "section": self.section,
            "text": self.text,
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "token_count": self.token_count,
            "chunk_type": self.chunk_type,
        }


@dataclass(frozen=True)
class SourceCitation:
    document_id: uuid.UUID
    external_id: str
    source_document: str
    source_uri: str
    document_type: str
    page_number: int | None
    section: str | None
    hospital: str | None
    physician: str | None
    visit_id: str | None
    checksum_sha256: str
    citation_label: str
    patient_ref: str | None = None
    diagnosis: str | None = None
    icd_codes: list[str] = field(default_factory=list)
    visit_date: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "external_id": self.external_id,
            "source_document": self.source_document,
            "source_uri": self.source_uri,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "section": self.section,
            "hospital": self.hospital,
            "physician": self.physician,
            "visit_id": self.visit_id,
            "checksum_sha256": self.checksum_sha256,
            "citation_label": self.citation_label,
            "patient_ref": self.patient_ref,
            "diagnosis": self.diagnosis,
            "icd_codes": self.icd_codes,
            "visit_date": self.visit_date,
        }


@dataclass(frozen=True)
class RetrievalConfidence:
    score: float
    level: str
    reranker_score: float | None
    hybrid_score: float
    ocr_confidence: float | None
    source_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "reranker_score": self.reranker_score,
            "hybrid_score": self.hybrid_score,
            "ocr_confidence": self.ocr_confidence,
            "source_count": self.source_count,
        }


@dataclass(frozen=True)
class ContextualSearchHit:
    final_rank: int
    matched_chunk: ChunkContext
    parent_context: ChunkContext | None
    citation: SourceCitation
    confidence: RetrievalConfidence
    retrieval: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "final_rank": self.final_rank,
            "matched_chunk": self.matched_chunk.to_metadata(),
            "parent_context": self.parent_context.to_metadata()
            if self.parent_context is not None
            else None,
            "citation": self.citation.to_metadata(),
            "confidence": self.confidence.to_metadata(),
            "retrieval": self.retrieval,
        }


@dataclass(frozen=True)
class ContextualSearchResult:
    query: str
    authorization: AuthorizedMetadataFilter
    hits: list[ContextualSearchHit]
    rerank_result: CrossEncoderRerankResult

    def to_metadata(self) -> dict[str, Any]:
        from app.services.timeline_reconstruction import reconstruct_patient_timeline

        timeline = reconstruct_patient_timeline(self)
        return {
            "retriever": CONTEXT_EXPANSION_VERSION,
            "query": self.query,
            "authorization": self.authorization.to_metadata(),
            "hit_count": len(self.hits),
            "reranker_model": self.rerank_result.reranker_model,
            "hits": [hit.to_metadata() for hit in self.hits],
            "timeline": [group.to_metadata() for group in timeline],
        }


def contextual_search(
    db: Session,
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
) -> ContextualSearchResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    rerank_result = reranked_hybrid_search(
        db,
        user=user,
        query=parsed_query,
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
    return expand_reranked_result(db, rerank_result=rerank_result)


def expand_reranked_result(
    db: Session,
    *,
    rerank_result: CrossEncoderRerankResult,
) -> ContextualSearchResult:
    hits = expand_reranked_hits(db, hits=rerank_result.hits)
    return ContextualSearchResult(
        query=rerank_result.query,
        authorization=rerank_result.authorization,
        hits=hits,
        rerank_result=rerank_result,
    )


def expand_reranked_hits(
    db: Session,
    *,
    hits: list[RerankedSearchHit],
) -> list[ContextualSearchHit]:
    chunks_by_id = fetch_chunks_for_context(db, [hit.chunk_id for hit in hits])
    expanded_hits: list[ContextualSearchHit] = []
    for hit in hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            continue
        parent_chunk = parent_context_chunk(chunk)
        document = chunk.document
        expanded_hits.append(
            ContextualSearchHit(
                final_rank=hit.final_rank,
                matched_chunk=chunk_context(chunk),
                parent_context=chunk_context(parent_chunk) if parent_chunk is not None else None,
                citation=source_citation(document=document, chunk=chunk, parent_chunk=parent_chunk),
                confidence=retrieval_confidence(hit=hit, chunk=chunk, document=document),
                retrieval=retrieval_metadata(hit),
            )
        )
    return deduplicate_contextual_hits(expanded_hits)


def deduplicate_contextual_hits(hits: list[ContextualSearchHit]) -> list[ContextualSearchHit]:
    deduplicated_hits: list[ContextualSearchHit] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for hit in hits:
        key = contextual_hit_dedupe_key(hit)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated_hits.append(replace(hit, final_rank=len(deduplicated_hits) + 1))
    return deduplicated_hits


def contextual_hit_dedupe_key(hit: ContextualSearchHit) -> tuple[str, str, str]:
    return (
        str(hit.citation.document_id),
        normalized_dedupe_text(hit.citation.section or hit.matched_chunk.section),
        normalized_dedupe_text(hit.matched_chunk.text),
    )


def normalized_dedupe_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def fetch_chunks_for_context(
    db: Session,
    chunk_ids: list[uuid.UUID],
) -> dict[uuid.UUID, DocumentChunk]:
    if not chunk_ids:
        return {}
    chunks = db.scalars(
        select(DocumentChunk)
        .options(
            selectinload(DocumentChunk.document),
            selectinload(DocumentChunk.parent_chunk),
        )
        .where(DocumentChunk.id.in_(chunk_ids))
    ).all()
    return {chunk.id: chunk for chunk in chunks}


def parent_context_chunk(chunk: DocumentChunk) -> DocumentChunk | None:
    if chunk.parent_chunk is not None:
        return chunk.parent_chunk
    metadata = chunk.retrieval_metadata or {}
    if metadata.get("chunk_type") == "parent":
        return chunk
    return None


def chunk_context(chunk: DocumentChunk) -> ChunkContext:
    metadata = chunk.retrieval_metadata or {}
    return ChunkContext(
        chunk_id=chunk.id,
        section=chunk.section,
        text=chunk.content,
        page_number=chunk.page_number,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        token_count=chunk.token_count,
        chunk_type=metadata.get("chunk_type"),
    )


def source_citation(
    *,
    document: Document,
    chunk: DocumentChunk,
    parent_chunk: DocumentChunk | None,
) -> SourceCitation:
    metadata = merged_retrieval_metadata(chunk=chunk, parent_chunk=parent_chunk)
    page_number = (
        chunk.page_number
        if chunk.page_number is not None
        else parent_chunk.page_number
        if parent_chunk
        else None
    )
    section = chunk.section or (parent_chunk.section if parent_chunk else None)
    return SourceCitation(
        document_id=document.id,
        external_id=document.external_id,
        source_document=document.file_name,
        source_uri=document.source_uri,
        document_type=document.document_type.value,
        page_number=page_number,
        section=section,
        hospital=document.hospital,
        physician=document.physician,
        visit_id=document.visit_id,
        checksum_sha256=document.checksum_sha256,
        citation_label=citation_label(
            external_id=document.external_id,
            source_document=document.file_name,
            page_number=page_number,
            section=section,
        ),
        patient_ref=str(metadata.get("patient_ref") or document.patient_ref),
        diagnosis=document.diagnosis or string_value(metadata.get("diagnosis")),
        icd_codes=citation_icd_codes(document=document, metadata=metadata),
        visit_date=citation_visit_date(document=document, metadata=metadata),
    )


def citation_label(
    *,
    external_id: str,
    source_document: str,
    page_number: int | None,
    section: str | None,
) -> str:
    parts = [external_id or source_document]
    if page_number is not None:
        parts.append(f"p. {page_number}")
    if section:
        parts.append(section)
    return " | ".join(parts)


def merged_retrieval_metadata(
    *,
    chunk: DocumentChunk,
    parent_chunk: DocumentChunk | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if parent_chunk is not None:
        metadata.update(parent_chunk.retrieval_metadata or {})
    metadata.update(chunk.retrieval_metadata or {})
    return metadata


def citation_icd_codes(
    *,
    document: Document,
    metadata: dict[str, Any],
) -> list[str]:
    if document.icd_codes:
        return [str(code) for code in document.icd_codes]

    clinical_entities = metadata.get("clinical_entities") or {}
    values = metadata.get("icd_codes")
    if not values and isinstance(clinical_entities, dict):
        values = clinical_entities.get("icd_codes")
    return list_values(values)


def citation_visit_date(
    *,
    document: Document,
    metadata: dict[str, Any],
) -> str | None:
    for candidate in (
        first_visit_date_from_metadata(metadata),
        first_visit_date_from_metadata(document.document_metadata or {}),
    ):
        if candidate:
            return candidate
    return None


def first_visit_date_from_metadata(metadata: dict[str, Any]) -> str | None:
    if metadata.get("visit_date"):
        return str(metadata["visit_date"])

    clinical_entities = metadata.get("clinical_entities") or {}
    if isinstance(clinical_entities, dict):
        date_value = first_visit_date_value(clinical_entities.get("dates"))
        if date_value:
            return date_value

    date_value = first_visit_date_value(metadata.get("dates"))
    if date_value:
        return date_value
    return None


def first_visit_date_value(dates: Any) -> str | None:
    if not isinstance(dates, list):
        return None
    for date_entry in dates:
        if not isinstance(date_entry, dict) or not date_entry.get("value"):
            continue
        if date_entry.get("label") in {None, "visit_date"}:
            return str(date_entry["value"])
    return None


def string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def retrieval_confidence(
    *,
    hit: RerankedSearchHit,
    chunk: DocumentChunk,
    document: Document,
) -> RetrievalConfidence:
    ocr_confidence = first_float(chunk.ocr_confidence, document.ocr_confidence)
    semantic_score = semantic_confidence(hit)
    confidence_score = semantic_score
    if ocr_confidence is not None:
        confidence_score = (semantic_score * 0.85) + (ocr_confidence * 0.15)
    confidence_score = round(clamp(confidence_score), 6)
    return RetrievalConfidence(
        score=confidence_score,
        level=confidence_level(confidence_score),
        reranker_score=hit.reranker_score,
        hybrid_score=hit.hybrid_score,
        ocr_confidence=ocr_confidence,
        source_count=len(hit.sources),
    )


def semantic_confidence(hit: RerankedSearchHit) -> float:
    if hit.reranker_score is not None:
        if 0 <= hit.reranker_score <= 1:
            return hit.reranker_score
        return 1 / (1 + math.exp(-hit.reranker_score))

    source_bonus = 0.1 if len(hit.sources) > 1 else 0.0
    return clamp(0.55 + min(hit.hybrid_score * 8, 0.3) + source_bonus)


def confidence_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
    return None


def clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def retrieval_metadata(hit: RerankedSearchHit) -> dict[str, Any]:
    return {
        "sources": hit.sources,
        "final_rank": hit.final_rank,
        "hybrid_rank": hit.hybrid_rank,
        "hybrid_score": hit.hybrid_score,
        "reranker_score": hit.reranker_score,
        "source_metadata": hit.source_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reranked search with parent context and citations."
    )
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
        result = contextual_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            rerank_top_n=args.rerank_top_n,
            rrf_k=args.rrf_k,
            collection_name=args.collection,
        )
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
