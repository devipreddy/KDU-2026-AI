from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.enums import ChunkIndexingStatus, DocumentStatus, IngestionJobStatus
from app.models.ingestion_job import IngestionJob
from app.services.chroma_index import (
    build_bm25_document,
    build_chroma_upsert_payload,
    chroma_id_for_chunk,
    ensure_chroma_collection,
)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
MAX_INDEXING_ERROR_LENGTH = 2000
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = Lock()


class EmbeddingPipelineError(RuntimeError):
    """Raised when chunk embeddings cannot be generated or stored."""


class EmbeddingEncoder(Protocol):
    model_name: str

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector for each input text."""


@dataclass(frozen=True)
class EmbeddingPipelineResult:
    model_name: str
    collection_name: str
    indexed_chunk_count: int
    embedding_dimension: int
    embedding_ids: list[str]
    chunk_ids: list[str]
    indexed_at: datetime

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "collection_name": self.collection_name,
            "indexed_chunk_count": self.indexed_chunk_count,
            "embedding_dimension": self.embedding_dimension,
            "embedding_ids": self.embedding_ids,
            "chunk_ids": self.chunk_ids,
            "indexed_at": self.indexed_at.isoformat(),
        }


class SentenceTransformerEmbeddingEncoder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.model = load_sentence_transformer_model(self.model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings_to_float_lists(raw_embeddings)


def load_sentence_transformer_model(model_name: str) -> Any:
    cached_model = _MODEL_CACHE.get(model_name)
    if cached_model is not None:
        return cached_model

    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(model_name)
        if cached_model is not None:
            return cached_model

        model = build_sentence_transformer_model(model_name)
        _MODEL_CACHE[model_name] = model
        return model


def build_sentence_transformer_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingPipelineError(
            "sentence-transformers is not installed. Install backend dependencies before "
            "running embedding generation."
        ) from exc

    return SentenceTransformer(model_name)


def select_chunks_for_embedding(
    db: Session,
    *,
    document_id: uuid.UUID | None = None,
    limit: int = 100,
    reindex: bool = False,
    retry_failed: bool = False,
) -> list[DocumentChunk]:
    query = select(DocumentChunk).order_by(DocumentChunk.created_at, DocumentChunk.chunk_index)
    if document_id is not None:
        query = query.where(DocumentChunk.document_id == document_id)
    if not reindex:
        eligible_statuses = [ChunkIndexingStatus.PENDING.value]
        if retry_failed:
            eligible_statuses.append(ChunkIndexingStatus.FAILED.value)
        query = query.where(DocumentChunk.indexing_status.in_(eligible_statuses))
    return list(db.scalars(query.limit(limit)).all())


def index_pending_chunks(
    db: Session,
    *,
    limit: int = 100,
    document_id: uuid.UUID | None = None,
    reindex: bool = False,
    retry_failed: bool = False,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
) -> EmbeddingPipelineResult:
    chunks = select_chunks_for_embedding(
        db,
        document_id=document_id,
        limit=limit,
        reindex=reindex,
        retry_failed=retry_failed,
    )
    return index_chunks_with_embeddings(
        db,
        chunks=chunks,
        encoder=encoder,
        collection=collection,
        collection_name=collection_name,
    )


def index_chunks_with_embeddings(
    db: Session,
    *,
    chunks: list[DocumentChunk],
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
) -> EmbeddingPipelineResult:
    indexed_at = datetime.now(UTC)
    if not chunks:
        return EmbeddingPipelineResult(
            model_name=encoder.model_name if encoder else settings.embedding_model_name,
            collection_name=collection_name
            or getattr(collection, "name", settings.chroma_collection),
            indexed_chunk_count=0,
            embedding_dimension=0,
            embedding_ids=[],
            chunk_ids=[],
            indexed_at=indexed_at,
        )

    chunk_ids = [chunk.id for chunk in chunks]
    active_collection_name = collection_name or getattr(
        collection,
        "name",
        settings.chroma_collection,
    )
    mark_chunks_indexing(
        db,
        chunks=chunks,
        collection_name=active_collection_name,
        started_at=indexed_at,
    )

    try:
        active_encoder = encoder or SentenceTransformerEmbeddingEncoder()
        active_collection = collection or ensure_chroma_collection(collection_name=collection_name)
        active_collection_name = collection_name or getattr(
            active_collection,
            "name",
            settings.chroma_collection,
        )
        embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
        embeddings = active_encoder.encode_documents(embedding_texts)
        embedding_dimension = validate_embeddings(embeddings, expected_count=len(chunks))

        for chunk in chunks:
            chunk.embedding_id = chroma_id_for_chunk(chunk)
            chunk.embedding_collection = active_collection_name
            chunk.indexing_status = ChunkIndexingStatus.INDEXED.value
            chunk.indexing_error = None
            chunk.indexed_at = indexed_at
            chunk.retrieval_metadata = {
                **(chunk.retrieval_metadata or {}),
                "embedding": {
                    "model": active_encoder.model_name,
                    "dimension": embedding_dimension,
                    "indexed_at": indexed_at.isoformat(),
                    "collection": active_collection_name,
                },
                "indexing": {
                    "status": ChunkIndexingStatus.INDEXED.value,
                    "attempt": chunk.indexing_attempts,
                    "indexed_at": indexed_at.isoformat(),
                    "collection": active_collection_name,
                },
            }

        payload = build_chroma_upsert_payload(chunks, embeddings=embeddings)
        active_collection.upsert(**payload)
        update_document_indexing_summaries(
            db,
            document_ids={chunk.document_id for chunk in chunks},
            collection_name=active_collection_name,
            indexed_at=indexed_at,
        )
        db.commit()
    except Exception as exc:
        mark_chunks_indexing_failed(
            db,
            chunk_ids=chunk_ids,
            error=exc,
            collection_name=active_collection_name,
        )
        if isinstance(exc, EmbeddingPipelineError):
            raise
        raise EmbeddingPipelineError(f"Chunk indexing failed: {exc}") from exc

    return EmbeddingPipelineResult(
        model_name=active_encoder.model_name,
        collection_name=active_collection_name,
        indexed_chunk_count=len(chunks),
        embedding_dimension=embedding_dimension,
        embedding_ids=[chunk.embedding_id or chroma_id_for_chunk(chunk) for chunk in chunks],
        chunk_ids=[str(chunk.id) for chunk in chunks],
        indexed_at=indexed_at,
    )


def index_document_chunks(
    db: Session,
    *,
    document_id: uuid.UUID,
    ingestion_job: IngestionJob | None = None,
    limit: int = 1000,
    reindex: bool = False,
    retry_failed: bool = False,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
) -> EmbeddingPipelineResult:
    document = db.get(Document, document_id)
    if document is None:
        raise EmbeddingPipelineError("Document was not found for chunk indexing")

    started_at = datetime.now(UTC)
    if ingestion_job is not None:
        ingestion_job.status = IngestionJobStatus.RUNNING
        ingestion_job.stage = "indexing_chunks"
        ingestion_job.error_message = None
        ingestion_job.started_at = ingestion_job.started_at or started_at
        db.commit()

    try:
        result = index_pending_chunks(
            db,
            limit=limit,
            document_id=document_id,
            reindex=reindex,
            retry_failed=retry_failed,
            encoder=encoder,
            collection=collection,
            collection_name=collection_name,
        )
    except EmbeddingPipelineError as exc:
        failed_at = datetime.now(UTC)
        db.rollback()
        document = db.get(Document, document_id)
        if document is not None:
            document.document_metadata = {
                **(document.document_metadata or {}),
                "indexing": {
                    **((document.document_metadata or {}).get("indexing") or {}),
                    "status": ChunkIndexingStatus.FAILED.value,
                    "failed_at": failed_at.isoformat(),
                    "error": truncate_indexing_error(exc),
                },
            }
        if ingestion_job is not None:
            ingestion_job.status = IngestionJobStatus.FAILED
            ingestion_job.stage = "indexing_failed"
            ingestion_job.error_message = truncate_indexing_error(exc)
            ingestion_job.finished_at = failed_at
            ingestion_job.job_metadata = {
                **(ingestion_job.job_metadata or {}),
                "indexing": {
                    "status": ChunkIndexingStatus.FAILED.value,
                    "failed_at": failed_at.isoformat(),
                    "error": truncate_indexing_error(exc),
                },
            }
        db.commit()
        raise

    finished_at = result.indexed_at
    document = db.get(Document, document_id)
    if document is not None:
        existing_indexing = (document.document_metadata or {}).get("indexing") or {}
        document.document_metadata = {
            **(document.document_metadata or {}),
            "indexing": {
                **existing_indexing,
                "last_result": result.to_metadata(),
            },
        }
    if ingestion_job is not None:
        ingestion_job.status = IngestionJobStatus.SUCCEEDED
        ingestion_job.stage = "indexed" if result.indexed_chunk_count else "indexing_noop"
        ingestion_job.finished_at = finished_at
        ingestion_job.job_metadata = {
            **(ingestion_job.job_metadata or {}),
            "indexing": result.to_metadata(),
        }
    db.commit()
    return result


def mark_chunks_indexing(
    db: Session,
    *,
    chunks: list[DocumentChunk],
    collection_name: str,
    started_at: datetime,
) -> None:
    for chunk in chunks:
        attempt = (chunk.indexing_attempts or 0) + 1
        chunk.indexing_status = ChunkIndexingStatus.INDEXING.value
        chunk.indexing_attempts = attempt
        chunk.indexing_error = None
        chunk.retrieval_metadata = {
            **(chunk.retrieval_metadata or {}),
            "indexing": {
                "status": ChunkIndexingStatus.INDEXING.value,
                "attempt": attempt,
                "started_at": started_at.isoformat(),
                "collection": collection_name,
            },
        }
    update_document_indexing_summaries(
        db,
        document_ids={chunk.document_id for chunk in chunks},
        collection_name=collection_name,
        indexed_at=None,
    )
    db.commit()


def mark_chunks_indexing_failed(
    db: Session,
    *,
    chunk_ids: list[uuid.UUID],
    error: Exception,
    collection_name: str,
) -> None:
    db.rollback()
    failed_at = datetime.now(UTC)
    chunks = list(db.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))).all())
    error_message = truncate_indexing_error(error)
    for chunk in chunks:
        chunk.indexing_status = ChunkIndexingStatus.FAILED.value
        chunk.indexing_error = error_message
        chunk.retrieval_metadata = {
            **(chunk.retrieval_metadata or {}),
            "indexing": {
                "status": ChunkIndexingStatus.FAILED.value,
                "attempt": chunk.indexing_attempts,
                "failed_at": failed_at.isoformat(),
                "collection": collection_name,
                "error": error_message,
            },
        }
    update_document_indexing_summaries(
        db,
        document_ids={chunk.document_id for chunk in chunks},
        collection_name=collection_name,
        indexed_at=None,
    )
    db.commit()


def update_document_indexing_summaries(
    db: Session,
    *,
    document_ids: set[uuid.UUID],
    collection_name: str,
    indexed_at: datetime | None,
) -> None:
    if not document_ids:
        return

    documents = list(db.scalars(select(Document).where(Document.id.in_(document_ids))).all())
    for document in documents:
        chunks = list(
            db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            ).all()
        )
        status_counts = {
            ChunkIndexingStatus.PENDING.value: 0,
            ChunkIndexingStatus.INDEXING.value: 0,
            ChunkIndexingStatus.INDEXED.value: 0,
            ChunkIndexingStatus.FAILED.value: 0,
        }
        for chunk in chunks:
            status_counts[chunk.indexing_status] = status_counts.get(chunk.indexing_status, 0) + 1

        overall_status = document_indexing_status(status_counts=status_counts, total=len(chunks))
        summary = {
            "status": overall_status,
            "collection_name": collection_name,
            "chunk_count": len(chunks),
            "status_counts": status_counts,
            "indexed_chunk_ids": [
                str(chunk.id)
                for chunk in chunks
                if chunk.indexing_status == ChunkIndexingStatus.INDEXED.value
            ],
            "failed_chunk_ids": [
                str(chunk.id)
                for chunk in chunks
                if chunk.indexing_status == ChunkIndexingStatus.FAILED.value
            ],
        }
        if indexed_at is not None and overall_status == ChunkIndexingStatus.INDEXED.value:
            summary["indexed_at"] = indexed_at.isoformat()

        document.document_metadata = {
            **(document.document_metadata or {}),
            "indexing": {
                **((document.document_metadata or {}).get("indexing") or {}),
                **summary,
            },
        }
        if (
            overall_status == ChunkIndexingStatus.INDEXED.value
            and document.status != DocumentStatus.REVIEW_REQUIRED
        ):
            document.status = DocumentStatus.INDEXED
        elif document.status == DocumentStatus.INDEXED:
            document.status = DocumentStatus.PROCESSED


def document_indexing_status(*, status_counts: dict[str, int], total: int) -> str:
    if total == 0:
        return ChunkIndexingStatus.PENDING.value
    if status_counts.get(ChunkIndexingStatus.FAILED.value, 0):
        return ChunkIndexingStatus.FAILED.value
    if status_counts.get(ChunkIndexingStatus.INDEXING.value, 0):
        return ChunkIndexingStatus.INDEXING.value
    if status_counts.get(ChunkIndexingStatus.PENDING.value, 0):
        return ChunkIndexingStatus.PENDING.value
    return ChunkIndexingStatus.INDEXED.value


def truncate_indexing_error(error: Exception) -> str:
    return str(error)[:MAX_INDEXING_ERROR_LENGTH]


def build_embedding_text(chunk: DocumentChunk) -> str:
    retrieval_metadata = chunk.retrieval_metadata or {}
    parts = [
        f"Section: {chunk.section}" if chunk.section else "",
        f"Diagnosis: {retrieval_metadata.get('diagnosis')}"
        if retrieval_metadata.get("diagnosis")
        else "",
        f"Hospital: {retrieval_metadata.get('hospital')}"
        if retrieval_metadata.get("hospital")
        else "",
        f"Physician: {retrieval_metadata.get('physician')}"
        if retrieval_metadata.get("physician")
        else "",
        build_bm25_document(chunk),
    ]
    return "\n".join(part for part in parts if part)


def validate_embeddings(embeddings: list[list[float]], *, expected_count: int) -> int:
    if len(embeddings) != expected_count:
        raise EmbeddingPipelineError("Embedding count does not match chunk count")
    if not embeddings:
        return 0

    embedding_dimension = len(embeddings[0])
    if embedding_dimension == 0:
        raise EmbeddingPipelineError("Embedding vectors cannot be empty")

    for embedding in embeddings:
        if len(embedding) != embedding_dimension:
            raise EmbeddingPipelineError("Embedding vectors must have consistent dimensions")
    return embedding_dimension


def embeddings_to_float_lists(raw_embeddings: Any) -> list[list[float]]:
    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()
    return [[float(value) for value in embedding] for embedding in raw_embeddings]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BGE embeddings and upsert chunks to Chroma."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--document-id", type=uuid.UUID)
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as db:
        result = index_pending_chunks(
            db,
            limit=args.limit,
            document_id=args.document_id,
            collection_name=args.collection,
            reindex=args.reindex,
            retry_failed=args.retry_failed,
        )
    print(json.dumps(result.to_metadata(), sort_keys=True))


if __name__ == "__main__":
    main()
