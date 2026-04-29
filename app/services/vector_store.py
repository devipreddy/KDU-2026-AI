from __future__ import annotations

from typing import Any

import chromadb

from app.core.config import Settings
from app.services.types import ChunkPayload


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="content_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[ChunkPayload], embeddings: list[list[float]], file_name: str) -> None:
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                self._to_metadata(chunk=chunk, file_name=file_name)
                for chunk in chunks
            ],
        )

    def delete_file_chunks(self, file_id: str) -> None:
        self.collection.delete(where={"file_id": file_id})

    def query(self, embedding: list[float], top_k: int, file_id: str | None = None) -> list[dict[str, Any]]:
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
        }
        if file_id:
            query_kwargs["where"] = {"file_id": file_id}

        result = self.collection.query(**query_kwargs)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        matches = []
        for chunk_id, distance, document, metadata in zip(ids, distances, documents, metadatas, strict=False):
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "distance": float(distance) if distance is not None else 1.0,
                    "document": document,
                    "metadata": metadata or {},
                }
            )
        return matches

    def _to_metadata(self, chunk: ChunkPayload, file_name: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "file_id": chunk.file_id,
            "file_name": file_name,
            "chunk_index": chunk.chunk_index,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        if "page_numbers" in chunk.metadata:
            metadata["page_numbers"] = ",".join(str(value) for value in chunk.metadata["page_numbers"])
        if "sources" in chunk.metadata:
            metadata["sources"] = ",".join(str(value) for value in chunk.metadata["sources"])
        return metadata
