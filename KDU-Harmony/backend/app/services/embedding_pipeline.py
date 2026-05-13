from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document_chunk import DocumentChunk
from app.services.chroma_index import (
    build_bm25_document,
    build_chroma_upsert_payload,
    chroma_id_for_chunk,
    ensure_chroma_collection,
)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


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
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingPipelineError(
                "sentence-transformers is not installed. Install backend dependencies before "
                "running embedding generation."
            ) from exc

        self.model = SentenceTransformer(self.model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings_to_float_lists(raw_embeddings)


def select_chunks_for_embedding(
    db: Session,
    *,
    document_id: uuid.UUID | None = None,
    limit: int = 100,
    reindex: bool = False,
) -> list[DocumentChunk]:
    query = select(DocumentChunk).order_by(DocumentChunk.created_at, DocumentChunk.chunk_index)
    if document_id is not None:
        query = query.where(DocumentChunk.document_id == document_id)
    if not reindex:
        query = query.where(DocumentChunk.embedding_id.is_(None))
    return list(db.scalars(query.limit(limit)).all())


def index_pending_chunks(
    db: Session,
    *,
    limit: int = 100,
    document_id: uuid.UUID | None = None,
    reindex: bool = False,
    encoder: EmbeddingEncoder | None = None,
    collection: Any | None = None,
    collection_name: str | None = None,
) -> EmbeddingPipelineResult:
    chunks = select_chunks_for_embedding(
        db,
        document_id=document_id,
        limit=limit,
        reindex=reindex,
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
        chunk.retrieval_metadata = {
            **(chunk.retrieval_metadata or {}),
            "embedding": {
                "model": active_encoder.model_name,
                "dimension": embedding_dimension,
                "indexed_at": indexed_at.isoformat(),
                "collection": active_collection_name,
            },
        }

    payload = build_chroma_upsert_payload(chunks, embeddings=embeddings)
    active_collection.upsert(**payload)
    db.commit()

    return EmbeddingPipelineResult(
        model_name=active_encoder.model_name,
        collection_name=active_collection_name,
        indexed_chunk_count=len(chunks),
        embedding_dimension=embedding_dimension,
        embedding_ids=[chunk.embedding_id or chroma_id_for_chunk(chunk) for chunk in chunks],
        chunk_ids=[str(chunk.id) for chunk in chunks],
        indexed_at=indexed_at,
    )


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
        )
    print(json.dumps(result.to_metadata(), sort_keys=True))


if __name__ == "__main__":
    main()
