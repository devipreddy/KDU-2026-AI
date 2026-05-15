from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.models.enums import ChunkIndexingStatus
from app.models.user import User
from app.services.bm25_retrieval import chunk_matches_where, chunk_patient_ref
from app.services.chroma_index import ensure_chroma_collection
from app.services.clinical_metadata import ordered_unique
from app.services.embedding_pipeline import (
    EmbeddingEncoder,
    EmbeddingPipelineError,
    SentenceTransformerEmbeddingEncoder,
    validate_embeddings,
)
from app.services.query_understanding import QueryUnderstandingResult, understand_query
from app.services.retrieval_authorization import (
    AuthorizedMetadataFilter,
    build_authorized_metadata_filter,
)

DENSE_RETRIEVER_VERSION = "chroma_dense_retriever_v1"
DENSE_QUERY_CONTEXT_FIELDS = (
    "patient_ref",
    "hospital",
    "physician",
    "document_type",
    "diagnosis",
    "diagnosis_category",
    "icd_codes",
)
POST_FILTER_MULTIPLIER = 50
POST_FILTER_MIN_CANDIDATES = 100
POST_FILTER_MAX_CANDIDATES = 500


@dataclass(frozen=True)
class DenseSearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    patient_ref: str
    section: str | None
    score: float
    distance: float | None
    embedding_id: str
    embedding_collection: str | None
    snippet: str
    chroma_metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "patient_ref": self.patient_ref,
            "section": self.section,
            "score": self.score,
            "distance": self.distance,
            "embedding_id": self.embedding_id,
            "embedding_collection": self.embedding_collection,
            "snippet": self.snippet,
            "chroma_metadata": self.chroma_metadata,
        }


@dataclass(frozen=True)
class DenseSearchResult:
    query: str
    query_text: str
    query_embedding_model: str
    query_embedding_dimension: int
    authorization: AuthorizedMetadataFilter
    hits: list[DenseSearchHit]
    candidate_count: int
    collection_name: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retriever": DENSE_RETRIEVER_VERSION,
            "query": self.query,
            "query_text": self.query_text,
            "query_embedding_model": self.query_embedding_model,
            "query_embedding_dimension": self.query_embedding_dimension,
            "authorization": self.authorization.to_metadata(),
            "candidate_count": self.candidate_count,
            "collection_name": self.collection_name,
            "hits": [hit.to_metadata() for hit in self.hits],
        }


def dense_vector_search(
    db: Session,
    *,
    user: User,
    query: str | QueryUnderstandingResult,
    limit: int = 10,
    authorized_patient_refs: list[str] | None = None,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
    authorization: AuthorizedMetadataFilter | None = None,
) -> DenseSearchResult:
    parsed_query = understand_query(query) if isinstance(query, str) else query
    raw_query = parsed_query.original_query
    authorization = authorization or build_authorized_metadata_filter(
        db,
        user=user,
        query=parsed_query,
        authorized_patient_refs=authorized_patient_refs,
    )
    active_collection_name = collection_name or getattr(
        collection,
        "name",
        settings.chroma_collection,
    )
    query_text = build_dense_query_text(parsed_query)
    model_name = encoder.model_name if encoder is not None else settings.embedding_model_name

    if authorization.denied or limit <= 0 or not query_text:
        return DenseSearchResult(
            query=raw_query,
            query_text=query_text,
            query_embedding_model=model_name,
            query_embedding_dimension=0,
            authorization=authorization,
            hits=[],
            candidate_count=0,
            collection_name=active_collection_name,
        )

    if indexed_chunk_count(db) == 0:
        return DenseSearchResult(
            query=raw_query,
            query_text=query_text,
            query_embedding_model=model_name,
            query_embedding_dimension=0,
            authorization=authorization,
            hits=[],
            candidate_count=0,
            collection_name=active_collection_name,
        )

    active_encoder = encoder or SentenceTransformerEmbeddingEncoder()
    query_embedding = active_encoder.encode_documents([query_text])
    embedding_dimension = validate_embeddings(query_embedding, expected_count=1)
    active_collection = collection or ensure_chroma_collection(collection_name=collection_name)
    active_collection_name = collection_name or getattr(
        active_collection,
        "name",
        settings.chroma_collection,
    )

    chroma_where = chroma_compatible_where(authorization.chroma_where)
    raw_results = query_chroma_collection(
        active_collection,
        query_embedding=query_embedding[0],
        limit=expanded_query_limit(
            requested_limit=limit,
            original_where=authorization.chroma_where,
            chroma_where=chroma_where,
        ),
        where=chroma_where,
    )
    hits = hydrate_dense_hits(
        db,
        raw_results=raw_results,
        authorization=authorization,
        limit=limit,
    )

    return DenseSearchResult(
        query=raw_query,
        query_text=query_text,
        query_embedding_model=active_encoder.model_name,
        query_embedding_dimension=embedding_dimension,
        authorization=authorization,
        hits=hits,
        candidate_count=len(first_chroma_result_list(raw_results.get("ids"))),
        collection_name=active_collection_name,
    )


def chroma_compatible_where(where: dict[str, Any]) -> dict[str, Any]:
    if not where:
        return {}
    if "$and" in where:
        predicates = [
            predicate
            for raw_predicate in where["$and"]
            if (predicate := chroma_compatible_where(raw_predicate))
        ]
        if not predicates:
            return {}
        if len(predicates) == 1:
            return predicates[0]
        return {"$and": predicates}
    if "$or" in where:
        predicates = [
            predicate
            for raw_predicate in where["$or"]
            if (predicate := chroma_compatible_where(raw_predicate))
        ]
        if not predicates or len(predicates) != len(where["$or"]):
            return {}
        return {"$or": predicates}

    compatible: dict[str, Any] = {}
    for field, condition in where.items():
        if not is_chroma_compatible_condition(condition):
            continue
        compatible[field] = condition
    return compatible


def is_chroma_compatible_condition(condition: Any) -> bool:
    if not isinstance(condition, dict):
        return True
    for operator, expected in condition.items():
        if operator in {"$gt", "$gte", "$lt", "$lte"} and not isinstance(expected, int | float):
            return False
    return True


def expanded_query_limit(
    *,
    requested_limit: int,
    original_where: dict[str, Any],
    chroma_where: dict[str, Any],
) -> int:
    if original_where == chroma_where:
        return max(requested_limit, 1)
    return max(
        requested_limit,
        min(
            POST_FILTER_MAX_CANDIDATES,
            max(POST_FILTER_MIN_CANDIDATES, requested_limit * POST_FILTER_MULTIPLIER),
        ),
    )


def indexed_chunk_count(db: Session) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.indexing_status == ChunkIndexingStatus.INDEXED.value)
        )
        or 0
    )


def query_chroma_collection(
    collection: Any,
    *,
    query_embedding: list[float],
    limit: int,
    where: dict[str, Any],
) -> dict[str, Any]:
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": max(limit, 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    try:
        return collection.query(**query_kwargs)
    except Exception as exc:
        raise EmbeddingPipelineError(f"Dense Chroma query failed: {exc}") from exc


def hydrate_dense_hits(
    db: Session,
    *,
    raw_results: dict[str, Any],
    authorization: AuthorizedMetadataFilter,
    limit: int,
) -> list[DenseSearchHit]:
    if limit <= 0:
        return []

    chroma_ids = [str(item) for item in first_chroma_result_list(raw_results.get("ids"))]
    distances = first_chroma_result_list(raw_results.get("distances"))
    metadatas = first_chroma_result_list(raw_results.get("metadatas"))
    chunks_by_chroma_id = fetch_chunks_by_chroma_ids(db, chroma_ids)

    hits: list[DenseSearchHit] = []
    for index, chroma_id in enumerate(chroma_ids):
        chunk = chunks_by_chroma_id.get(chroma_id)
        if chunk is None:
            continue
        if chunk.indexing_status != ChunkIndexingStatus.INDEXED.value:
            continue
        if not chunk_matches_where(chunk, authorization.chroma_where):
            continue

        distance = distance_at(distances, index)
        metadata = metadata_at(metadatas, index)
        hits.append(
            DenseSearchHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                patient_ref=chunk_patient_ref(chunk),
                section=chunk.section,
                score=score_from_distance(distance),
                distance=distance,
                embedding_id=chunk.embedding_id or chroma_id,
                embedding_collection=chunk.embedding_collection,
                snippet=build_dense_snippet(chunk.content),
                chroma_metadata=metadata,
            )
        )
        if len(hits) >= limit:
            break
    return hits


def fetch_chunks_by_chroma_ids(
    db: Session,
    chroma_ids: list[str],
) -> dict[str, DocumentChunk]:
    if not chroma_ids:
        return {}

    chunk_ids = [
        chunk_id for chroma_id in chroma_ids if (chunk_id := parse_chroma_chunk_id(chroma_id))
    ]
    conditions = [DocumentChunk.embedding_id.in_(chroma_ids)]
    if chunk_ids:
        conditions.append(DocumentChunk.id.in_(chunk_ids))

    chunks = db.scalars(
        select(DocumentChunk).options(selectinload(DocumentChunk.document)).where(or_(*conditions))
    ).all()

    by_chroma_id: dict[str, DocumentChunk] = {}
    for chunk in chunks:
        if chunk.embedding_id:
            by_chroma_id[chunk.embedding_id] = chunk
        by_chroma_id[f"chunk:{chunk.id}"] = chunk
        by_chroma_id[str(chunk.id)] = chunk
    return by_chroma_id


def parse_chroma_chunk_id(chroma_id: str) -> uuid.UUID | None:
    raw_id = chroma_id.removeprefix("chunk:")
    try:
        return uuid.UUID(raw_id)
    except ValueError:
        return None


def build_dense_query_text(query: QueryUnderstandingResult) -> str:
    parts = [query.normalized_query]

    diagnosis_values = [
        diagnosis
        for concept in query.diagnosis_concepts
        for diagnosis in [concept.concept, *concept.diagnoses]
    ]
    if diagnosis_values:
        parts.append("Diagnoses: " + "; ".join(ordered_unique(diagnosis_values)))

    for field in DENSE_QUERY_CONTEXT_FIELDS:
        values = normalize_context_values(query.metadata_filters.get(field))
        if values:
            parts.append(f"{field.replace('_', ' ').title()}: {'; '.join(values)}")

    temporal_values = normalize_temporal_filters(query.metadata_filters.get("visit_date"))
    if temporal_values:
        parts.append("Visit Dates: " + "; ".join(temporal_values))
    return "\n".join(part for part in parts if part).strip()


def normalize_context_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return ordered_unique(str(item) for item in value if item is not None and str(item))
    return [str(value)]


def normalize_temporal_filters(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    ranges: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start_date = item.get("start_date")
        end_date = item.get("end_date")
        matched_text = item.get("matched_text")
        if start_date and end_date:
            ranges.append(f"{start_date} to {end_date}")
        elif start_date:
            ranges.append(f"from {start_date}")
        elif end_date:
            ranges.append(f"through {end_date}")
        elif matched_text:
            ranges.append(str(matched_text))
    return ordered_unique(ranges)


def first_chroma_result_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        if value and isinstance(value[0], list | tuple):
            return list(value[0])
        return list(value)
    return []


def distance_at(distances: list[Any], index: int) -> float | None:
    try:
        distance = distances[index]
    except IndexError:
        return None
    if distance is None:
        return None
    return float(distance)


def metadata_at(metadatas: list[Any], index: int) -> dict[str, Any]:
    try:
        metadata = metadatas[index]
    except IndexError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def score_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return round(1 / (1 + max(distance, 0.0)), 6)


def build_dense_snippet(content: str, *, max_length: int = 260) -> str:
    return content[:max_length].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run authorization-aware dense vector retrieval.")
    parser.add_argument("email")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
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
        result = dense_vector_search(
            db,
            user=user,
            query=" ".join(args.query),
            limit=args.limit,
            collection_name=args.collection,
        )
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
