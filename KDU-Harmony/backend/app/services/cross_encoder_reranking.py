from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.models.user import User
from app.services.dense_retrieval import build_dense_query_text
from app.services.embedding_pipeline import EmbeddingEncoder
from app.services.hybrid_retrieval import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_RRF_K,
    HybridSearchHit,
    HybridSearchResult,
    hybrid_search,
)
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import AuthorizedMetadataFilter

RERANKER_VERSION = "cross_encoder_reranker_v1"
DEFAULT_RERANK_TOP_N = 8
_CROSS_ENCODER_CACHE: dict[tuple[str, str], tuple[Any, str, bool]] = {}
_CROSS_ENCODER_CACHE_LOCK = Lock()


class RerankingError(RuntimeError):
    """Raised when cross-encoder reranking cannot complete."""


class CrossEncoderReranker(Protocol):
    model_name: str

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score for each query/document pair."""


@dataclass(frozen=True)
class RerankedSearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_ref: str
    section: str | None
    final_rank: int
    hybrid_rank: int
    hybrid_score: float
    reranker_score: float | None
    sources: list[str]
    snippet: str
    source_metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "patient_ref": self.patient_ref,
            "section": self.section,
            "final_rank": self.final_rank,
            "hybrid_rank": self.hybrid_rank,
            "hybrid_score": self.hybrid_score,
            "reranker_score": self.reranker_score,
            "sources": self.sources,
            "snippet": self.snippet,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class CrossEncoderRerankResult:
    query: str
    authorization: AuthorizedMetadataFilter
    hits: list[RerankedSearchHit]
    hybrid_result: HybridSearchResult
    reranker_model: str
    rerank_top_n: int
    reranked_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retriever": RERANKER_VERSION,
            "query": self.query,
            "reranker_model": self.reranker_model,
            "rerank_top_n": self.rerank_top_n,
            "reranked_count": self.reranked_count,
            "authorization": self.authorization.to_metadata(),
            "hybrid_candidate_count": len(self.hybrid_result.hits),
            "hits": [hit.to_metadata() for hit in self.hits],
        }


class SentenceTransformersCrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
    ) -> None:
        preferred_model = model_name or settings.reranker_model_name
        fallback_model = fallback_model_name or settings.reranker_fallback_model_name
        self.model, self.model_name, self.fallback_used = load_cross_encoder_model(
            preferred_model,
            fallback_model or "",
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw_scores = self.model.predict(
            pairs,
            batch_size=settings.reranker_batch_size,
            show_progress_bar=False,
        )
        return scores_to_float_list(raw_scores, expected_count=len(pairs))


def load_cross_encoder_model(preferred_model: str, fallback_model: str) -> tuple[Any, str, bool]:
    cache_key = (preferred_model, fallback_model)
    cached_model = _CROSS_ENCODER_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    with _CROSS_ENCODER_CACHE_LOCK:
        cached_model = _CROSS_ENCODER_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        model = build_cross_encoder_model(preferred_model, fallback_model)
        _CROSS_ENCODER_CACHE[cache_key] = model
        return model


def build_cross_encoder_model(preferred_model: str, fallback_model: str) -> tuple[Any, str, bool]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankingError(
            "sentence-transformers is not installed. Install backend dependencies before "
            "running cross-encoder reranking."
        ) from exc

    try:
        return CrossEncoder(preferred_model), preferred_model, False
    except Exception as preferred_exc:
        if not fallback_model or fallback_model == preferred_model:
            raise RerankingError(
                f"Cross-encoder model failed to load: {preferred_model}"
            ) from preferred_exc
        try:
            return CrossEncoder(fallback_model), fallback_model, True
        except Exception as fallback_exc:
            raise RerankingError(
                f"Cross-encoder models failed to load: {preferred_model}, fallback {fallback_model}"
            ) from fallback_exc


def reranked_hybrid_search(
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
) -> CrossEncoderRerankResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    hybrid_limit = max(limit, rerank_top_n, 0)
    hybrid_result = hybrid_search(
        db,
        user=user,
        query=parsed_query,
        limit=hybrid_limit,
        candidate_limit=max(candidate_limit, rerank_top_n),
        rrf_k=rrf_k,
        authorized_patient_refs=authorized_patient_refs,
        encoder=encoder,
        collection=collection,
        collection_name=collection_name,
    )
    active_model_name = reranker.model_name if reranker else settings.reranker_model_name

    if limit <= 0 or rerank_top_n <= 0 or not hybrid_result.hits:
        return CrossEncoderRerankResult(
            query=parsed_query.original_query,
            authorization=hybrid_result.authorization,
            hits=hybrid_hits_to_reranked_hits(hybrid_result.hits, limit=limit),
            hybrid_result=hybrid_result,
            reranker_model=active_model_name,
            rerank_top_n=max(rerank_top_n, 0),
            reranked_count=0,
        )

    active_reranker = reranker or SentenceTransformersCrossEncoderReranker()
    candidate_count = min(rerank_top_n, len(hybrid_result.hits))
    top_candidates = hybrid_result.hits[:candidate_count]
    tail_candidates = hybrid_result.hits[candidate_count:]
    query_text = build_dense_query_text(parsed_query)
    chunks_by_id = fetch_chunks_for_reranking(db, top_candidates)
    pairs = [
        (query_text, build_reranker_document_text(hit, chunks_by_id.get(hit.chunk_id)))
        for hit in top_candidates
    ]
    reranker_scores = active_reranker.score_pairs(pairs)
    reranked_candidates = sort_reranked_candidates(
        candidates=top_candidates,
        scores=reranker_scores,
    )
    final_hybrid_hits = [*reranked_candidates, *tail_candidates]

    return CrossEncoderRerankResult(
        query=parsed_query.original_query,
        authorization=hybrid_result.authorization,
        hits=hybrid_hits_to_reranked_hits(
            final_hybrid_hits,
            limit=limit,
            reranker_scores_by_chunk={
                hit.chunk_id: score
                for hit, score in zip(top_candidates, reranker_scores, strict=True)
            },
            hybrid_ranks_by_chunk={
                hit.chunk_id: rank for rank, hit in enumerate(hybrid_result.hits, start=1)
            },
        ),
        hybrid_result=hybrid_result,
        reranker_model=active_reranker.model_name,
        rerank_top_n=rerank_top_n,
        reranked_count=candidate_count,
    )


def sort_reranked_candidates(
    *,
    candidates: list[HybridSearchHit],
    scores: list[float],
) -> list[HybridSearchHit]:
    if len(candidates) != len(scores):
        raise RerankingError("Reranker score count does not match candidate count")
    ranked = [
        (candidate, score, rank)
        for rank, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1)
    ]
    return [
        candidate
        for candidate, _, _ in sorted(
            ranked,
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[2],
                str(item[0].document_id),
                str(item[0].chunk_id),
            ),
        )
    ]


def hybrid_hits_to_reranked_hits(
    hits: list[HybridSearchHit],
    *,
    limit: int,
    reranker_scores_by_chunk: dict[uuid.UUID, float] | None = None,
    hybrid_ranks_by_chunk: dict[uuid.UUID, int] | None = None,
) -> list[RerankedSearchHit]:
    if limit <= 0:
        return []
    reranker_scores_by_chunk = reranker_scores_by_chunk or {}
    hybrid_ranks_by_chunk = hybrid_ranks_by_chunk or {
        hit.chunk_id: rank for rank, hit in enumerate(hits, start=1)
    }
    return [
        RerankedSearchHit(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            patient_ref=hit.patient_ref,
            section=hit.section,
            final_rank=final_rank,
            hybrid_rank=hybrid_ranks_by_chunk[hit.chunk_id],
            hybrid_score=hit.score,
            reranker_score=reranker_scores_by_chunk.get(hit.chunk_id),
            sources=hit.sources,
            snippet=hit.snippet,
            source_metadata=hit.source_metadata,
        )
        for final_rank, hit in enumerate(hits[:limit], start=1)
    ]


def fetch_chunks_for_reranking(
    db: Session,
    hits: list[HybridSearchHit],
) -> dict[uuid.UUID, DocumentChunk]:
    chunk_ids = [hit.chunk_id for hit in hits]
    if not chunk_ids:
        return {}
    chunks = db.scalars(
        select(DocumentChunk)
        .options(selectinload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    ).all()
    return {chunk.id: chunk for chunk in chunks}


def build_reranker_document_text(
    hit: HybridSearchHit,
    chunk: DocumentChunk | None,
) -> str:
    if chunk is None:
        return hit.snippet

    metadata = chunk.retrieval_metadata or {}
    clinical_entities = metadata.get("clinical_entities") or {}
    lexical_matches = lexical_match_summary(hit.source_metadata)
    parts = [
        f"Hybrid sources: {', '.join(hit.sources)}" if hit.sources else "",
        f"Lexical matches: {lexical_matches}" if lexical_matches else "",
        f"Matched snippet: {hit.snippet}" if hit.snippet else "",
        f"Section: {chunk.section}" if chunk.section else "",
        f"Document type: {metadata.get('document_type')}" if metadata.get("document_type") else "",
        f"Diagnosis: {metadata.get('diagnosis')}" if metadata.get("diagnosis") else "",
        f"ICD codes: {join_reranker_values(metadata.get('icd_codes'))}"
        if metadata.get("icd_codes")
        else "",
        f"Hospital: {metadata.get('hospital')}" if metadata.get("hospital") else "",
        f"Physician: {metadata.get('physician')}" if metadata.get("physician") else "",
    ]
    if isinstance(clinical_entities, dict):
        parts.extend(
            [
                f"Medications: {join_reranker_values(clinical_entities.get('medications'))}",
                f"Symptoms: {join_reranker_values(clinical_entities.get('symptoms'))}",
            ]
        )
    parts.append(chunk.content)
    return "\n".join(part for part in parts if part)


def lexical_match_summary(source_metadata: dict[str, Any]) -> str:
    bm25_metadata = source_metadata.get("bm25") if isinstance(source_metadata, dict) else None
    if not isinstance(bm25_metadata, dict):
        return ""

    matched_fields = bm25_metadata.get("matched_fields")
    if not isinstance(matched_fields, dict):
        return ""

    parts = []
    for field_name, values in matched_fields.items():
        rendered_values = join_reranker_values(values)
        if rendered_values:
            parts.append(f"{field_name.replace('_', ' ')}: {rendered_values}")
    return "; ".join(parts)


def join_reranker_values(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple | set | frozenset):
        return "; ".join(str(item) for item in value if item is not None and str(item))
    return str(value)


def scores_to_float_list(raw_scores: Any, *, expected_count: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    scores = [float(score) for score in raw_scores]
    if len(scores) != expected_count:
        raise RerankingError("Reranker score count does not match candidate count")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-encoder reranking over hybrid results.")
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
        result = reranked_hybrid_search(
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
