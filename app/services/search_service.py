from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.database import ApiUsageRecord, ChunkRecord, FileRecord
from app.schemas.search import SearchHitResponse, SearchResponse
from app.services.embedding_service import EmbeddingService
from app.services.presenters import usage_to_record_payload
from app.services.types import ApiUsageEntry
from app.services.vector_store import VectorStore


class SearchService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(self, db: Session, query: str, file_id: str | None, top_k: int) -> SearchResponse:
        query_embedding, usage = self.embedding_service.embed_query(query)
        matches = self.vector_store.query(query_embedding, top_k=top_k, file_id=file_id)
        self._log_usage(db, usage, file_id=file_id, matches=matches)
        if not matches:
            return SearchResponse(query=query, hits=[])

        chunk_ids = [match["chunk_id"] for match in matches]
        chunk_records = db.scalars(
            select(ChunkRecord)
            .where(ChunkRecord.id.in_(chunk_ids))
            .options(selectinload(ChunkRecord.file))
        ).all()
        chunk_map = {chunk.id: chunk for chunk in chunk_records}

        hits: list[SearchHitResponse] = []
        for match in matches:
            chunk = chunk_map.get(match["chunk_id"])
            if not chunk or not chunk.file:
                continue
            context = self._build_context(db, chunk)
            metadata = match["metadata"] or {}
            page_numbers = chunk.metadata_json.get("page_numbers", [])
            hits.append(
                SearchHitResponse(
                    chunk_id=chunk.id,
                    file_id=chunk.file_id,
                    file_name=chunk.file.file_name,
                    page_number=chunk.page_number,
                    score=round(max(0.0, 1.0 - float(match["distance"])), 4),
                    content=chunk.content,
                    context=context,
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "page_numbers": page_numbers,
                    },
                )
            )
        return SearchResponse(query=query, hits=hits)

    def _build_context(self, db: Session, chunk: ChunkRecord) -> str:
        previous_chunk = db.scalar(
            select(ChunkRecord)
            .where(ChunkRecord.file_id == chunk.file_id, ChunkRecord.chunk_index == chunk.chunk_index - 1)
        )
        next_chunk = db.scalar(
            select(ChunkRecord)
            .where(ChunkRecord.file_id == chunk.file_id, ChunkRecord.chunk_index == chunk.chunk_index + 1)
        )
        parts = []
        if previous_chunk:
            parts.append(previous_chunk.content)
        parts.append(chunk.content)
        if next_chunk:
            parts.append(next_chunk.content)
        return "\n\n".join(parts).strip()

    def _log_usage(
        self,
        db: Session,
        usage: ApiUsageEntry,
        file_id: str | None,
        matches: list[dict],
    ) -> None:
        attributed_file_ids = self._resolve_attributed_file_ids(file_id, matches)
        if not attributed_file_ids:
            db.add(ApiUsageRecord(**usage_to_record_payload(usage, file_id=None)))
            db.commit()
            return

        if len(attributed_file_ids) == 1:
            entry = ApiUsageEntry(
                operation=usage.operation,
                provider=usage.provider,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=usage.estimated_cost_usd,
                response_ms=usage.response_ms,
                metadata={**usage.metadata, "attribution_scope": "single_file"},
            )
            db.add(ApiUsageRecord(**usage_to_record_payload(entry, file_id=attributed_file_ids[0])))
            db.commit()
            return

        input_parts = self._split_integer(usage.input_tokens, len(attributed_file_ids))
        output_parts = self._split_integer(usage.output_tokens, len(attributed_file_ids))
        cost_parts = self._split_cost(usage.estimated_cost_usd, len(attributed_file_ids))
        for index, attributed_file_id in enumerate(attributed_file_ids):
            input_tokens = input_parts[index]
            output_tokens = output_parts[index]
            entry = ApiUsageEntry(
                operation=usage.operation,
                provider=usage.provider,
                model=usage.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                estimated_cost_usd=cost_parts[index],
                response_ms=usage.response_ms,
                metadata={
                    **usage.metadata,
                    "attribution_scope": "matched_files",
                    "attributed_file_ids": attributed_file_ids,
                    "attribution_index": index,
                    "shared_query_cost": True,
                },
            )
            db.add(ApiUsageRecord(**usage_to_record_payload(entry, file_id=attributed_file_id)))
        db.commit()

    def _resolve_attributed_file_ids(self, file_id: str | None, matches: list[dict]) -> list[str]:
        if file_id:
            return [file_id]

        ordered: list[str] = []
        seen: set[str] = set()
        for match in matches:
            metadata = match.get("metadata") or {}
            candidate = metadata.get("file_id")
            if isinstance(candidate, str) and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def _split_integer(self, value: int, parts: int) -> list[int]:
        base = value // parts
        remainder = value % parts
        return [base + (1 if index < remainder else 0) for index in range(parts)]

    def _split_cost(self, value: float, parts: int) -> list[float]:
        base = round(value / parts, 8)
        output = [base for _ in range(parts)]
        running_total = round(sum(output), 8)
        delta = round(value - running_total, 8)
        if output:
            output[-1] = round(output[-1] + delta, 8)
        return output
