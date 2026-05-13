from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.user import User
from app.services.bm25_retrieval import BM25SearchHit, BM25SearchResult, bm25_search
from app.services.dense_retrieval import DenseSearchHit, DenseSearchResult, dense_vector_search
from app.services.embedding_pipeline import EmbeddingEncoder
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import AuthorizedMetadataFilter

HYBRID_RETRIEVER_VERSION = "rrf_hybrid_retriever_v1"
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_LIMIT = 50


@dataclass(frozen=True)
class RRFContribution:
    retriever: str
    rank: int
    original_score: float
    rrf_score: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retriever": self.retriever,
            "rank": self.rank,
            "original_score": self.original_score,
            "rrf_score": self.rrf_score,
        }


@dataclass(frozen=True)
class HybridSearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_ref: str
    section: str | None
    score: float
    best_rank: int
    sources: list[str]
    contributions: list[RRFContribution]
    bm25_score: float | None
    dense_score: float | None
    snippet: str
    source_metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "patient_ref": self.patient_ref,
            "section": self.section,
            "score": self.score,
            "best_rank": self.best_rank,
            "sources": self.sources,
            "contributions": [contribution.to_metadata() for contribution in self.contributions],
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "snippet": self.snippet,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class HybridSearchResult:
    query: str
    authorization: AuthorizedMetadataFilter
    hits: list[HybridSearchHit]
    bm25_result: BM25SearchResult
    dense_result: DenseSearchResult
    rrf_k: int
    candidate_limit: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retriever": HYBRID_RETRIEVER_VERSION,
            "query": self.query,
            "rrf_k": self.rrf_k,
            "candidate_limit": self.candidate_limit,
            "authorization": self.authorization.to_metadata(),
            "bm25_hit_count": len(self.bm25_result.hits),
            "dense_hit_count": len(self.dense_result.hits),
            "bm25_candidate_count": self.bm25_result.candidate_chunk_count,
            "dense_candidate_count": self.dense_result.candidate_count,
            "hits": [hit.to_metadata() for hit in self.hits],
        }


def hybrid_search(
    db: Session,
    *,
    user: User,
    query: str | QueryUnderstandingResult,
    limit: int = 10,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    rrf_k: int = DEFAULT_RRF_K,
    authorized_patient_refs: list[str] | None = None,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
) -> HybridSearchResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    source_limit = max(limit, candidate_limit, 0)
    bm25_result = bm25_search(
        db,
        user=user,
        query=parsed_query,
        limit=source_limit,
        authorized_patient_refs=authorized_patient_refs,
    )
    dense_result = dense_vector_search(
        db,
        user=user,
        query=parsed_query,
        limit=source_limit,
        authorized_patient_refs=authorized_patient_refs,
        encoder=encoder,
        collection=collection,
        collection_name=collection_name,
    )
    hits = reciprocal_rank_fusion(
        bm25_hits=bm25_result.hits,
        dense_hits=dense_result.hits,
        limit=limit,
        rrf_k=rrf_k,
    )
    return HybridSearchResult(
        query=parsed_query.original_query,
        authorization=bm25_result.authorization,
        hits=hits,
        bm25_result=bm25_result,
        dense_result=dense_result,
        rrf_k=rrf_k,
        candidate_limit=source_limit,
    )


def reciprocal_rank_fusion(
    *,
    bm25_hits: list[BM25SearchHit],
    dense_hits: list[DenseSearchHit],
    limit: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchHit]:
    if limit <= 0:
        return []

    fused: dict[uuid.UUID, dict[str, Any]] = {}
    add_ranked_hits(fused, retriever="bm25", hits=bm25_hits, rrf_k=rrf_k)
    add_ranked_hits(fused, retriever="dense", hits=dense_hits, rrf_k=rrf_k)

    hits = [build_hybrid_hit(item) for item in fused.values()]
    return sorted(
        hits,
        key=lambda hit: (
            -hit.score,
            hit.best_rank,
            -len(hit.sources),
            str(hit.document_id),
            str(hit.chunk_id),
        ),
    )[:limit]


def add_ranked_hits(
    fused: dict[uuid.UUID, dict[str, Any]],
    *,
    retriever: str,
    hits: list[BM25SearchHit] | list[DenseSearchHit],
    rrf_k: int,
) -> None:
    seen_chunk_ids: set[uuid.UUID] = set()
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(hit.chunk_id)
        contribution = RRFContribution(
            retriever=retriever,
            rank=rank,
            original_score=hit.score,
            rrf_score=round(1 / (rrf_k + rank), 8),
        )
        accumulator = fused.setdefault(hit.chunk_id, new_accumulator(hit))
        accumulator["score"] += contribution.rrf_score
        accumulator["best_rank"] = min(accumulator["best_rank"], rank)
        accumulator["contributions"].append(contribution)
        accumulator["source_hits"][retriever] = hit


def new_accumulator(hit: BM25SearchHit | DenseSearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "patient_ref": hit.patient_ref,
        "section": hit.section,
        "score": 0.0,
        "best_rank": 10**9,
        "contributions": [],
        "source_hits": {},
    }


def build_hybrid_hit(accumulator: dict[str, Any]) -> HybridSearchHit:
    source_hits = accumulator["source_hits"]
    bm25_hit = source_hits.get("bm25")
    dense_hit = source_hits.get("dense")
    snippet_source = bm25_hit or dense_hit
    contributions = accumulator["contributions"]
    sources = [contribution.retriever for contribution in contributions]
    return HybridSearchHit(
        chunk_id=accumulator["chunk_id"],
        document_id=accumulator["document_id"],
        patient_ref=accumulator["patient_ref"],
        section=accumulator["section"],
        score=round(accumulator["score"], 8),
        best_rank=accumulator["best_rank"],
        sources=sources,
        contributions=contributions,
        bm25_score=bm25_hit.score if bm25_hit else None,
        dense_score=dense_hit.score if dense_hit else None,
        snippet=snippet_source.snippet if snippet_source else "",
        source_metadata=source_metadata(bm25_hit=bm25_hit, dense_hit=dense_hit),
    )


def source_metadata(
    *,
    bm25_hit: BM25SearchHit | None,
    dense_hit: DenseSearchHit | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if bm25_hit is not None:
        metadata["bm25"] = {
            "matched_fields": bm25_hit.matched_fields,
            "bm25_score": bm25_hit.bm25_score,
            "exact_match_score": bm25_hit.exact_match_score,
        }
    if dense_hit is not None:
        metadata["dense"] = {
            "distance": dense_hit.distance,
            "embedding_id": dense_hit.embedding_id,
            "embedding_collection": dense_hit.embedding_collection,
            "chroma_metadata": dense_hit.chroma_metadata,
        }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RRF hybrid retrieval over BM25 and dense hits."
    )
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
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
        result = hybrid_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
            collection_name=args.collection,
        )
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
