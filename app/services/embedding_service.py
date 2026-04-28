from __future__ import annotations

from itertools import islice

from app.services.providers.openai_provider import OpenAIProvider
from app.services.types import ApiUsageEntry, ChunkPayload
from app.services.vector_store import VectorStore


class EmbeddingService:
    def __init__(self, provider: OpenAIProvider, vector_store: VectorStore, batch_size: int = 32) -> None:
        self.provider = provider
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index_chunks(self, file_name: str, chunks: list[ChunkPayload]) -> list[ApiUsageEntry]:
        usage_entries: list[ApiUsageEntry] = []
        for batch in self._batched(chunks, self.batch_size):
            embeddings, usage = self.provider.create_embeddings(
                texts=[chunk.content for chunk in batch],
                operation="chunk_embeddings",
            )
            usage.metadata = {**usage.metadata, "chunk_ids": [chunk.chunk_id for chunk in batch], "chunk_count": len(batch)}
            self.vector_store.upsert_chunks(batch, embeddings, file_name=file_name)
            usage_entries.append(usage)
        return usage_entries

    def embed_query(self, query: str) -> tuple[list[float], ApiUsageEntry]:
        embeddings, usage = self.provider.create_embeddings(texts=[query], operation="query_embedding")
        return embeddings[0], usage

    def _batched(self, items: list[ChunkPayload], size: int):
        iterator = iter(items)
        while batch := list(islice(iterator, size)):
            yield batch
