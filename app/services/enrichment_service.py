from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.services.chunking import TokenChunker
from app.services.providers.openai_provider import OpenAIProvider
from app.services.types import EnrichmentResult


class EnrichmentService:
    def __init__(self, settings: Settings, provider: OpenAIProvider, chunker: TokenChunker) -> None:
        self.settings = settings
        self.provider = provider
        self.chunker = chunker

    def enrich(self, text: str) -> EnrichmentResult:
        token_count = self.chunker.count_tokens(text)
        if token_count <= self.settings.summary_max_input_tokens:
            payload, usage = self.provider.enrich_document(text=text, operation="document_enrichment")
            return self._normalize_payload(payload, [usage])

        chunk_texts = self.chunker.split_text(
            text=text,
            max_tokens=self.settings.summary_chunk_tokens,
            overlap_tokens=self.settings.summary_chunk_overlap_tokens,
        )
        usage_entries = []
        chunk_notes: list[str] = []
        total_chunks = len(chunk_texts)

        for index, chunk_text in enumerate(chunk_texts, start=1):
            payload, usage = self.provider.enrich_document(
                text=chunk_text,
                operation="enrichment_chunk",
                chunk_context=(
                    f"This is chunk {index} of {total_chunks} from a larger document. "
                    "Provide a short summary, up to 3 key_points, and up to 3 topic_tags."
                ),
            )
            usage_entries.append(usage)
            chunk_notes.append(self._chunk_note(payload, index))

        merged_payload, merged_usage = self.provider.enrich_document(
            text="\n\n".join(chunk_notes),
            operation="enrichment_merge",
            chunk_context=(
                "The following text contains chunk-level notes from a larger document. "
                "Merge them into a single cohesive title, 150-word summary, 5-7 key_points, and deduplicated topic_tags."
            ),
        )
        usage_entries.append(merged_usage)
        return self._normalize_payload(merged_payload, usage_entries)

    def _normalize_payload(self, payload: dict[str, Any], usage_entries) -> EnrichmentResult:
        title = self._coerce_text(payload.get("title"))
        summary = self._coerce_text(payload.get("summary")) or "Summary unavailable."
        key_points = self._coerce_string_list(payload.get("key_points"))
        topic_tags = self._dedupe(self._coerce_string_list(payload.get("topic_tags")))
        if not key_points:
            key_points = [summary]
        return EnrichmentResult(
            title=title,
            summary=summary,
            key_points=key_points[:7],
            topic_tags=topic_tags[:8],
            usage_entries=list(usage_entries),
        )

    def _chunk_note(self, payload: dict[str, Any], index: int) -> str:
        summary = self._coerce_text(payload.get("summary")) or ""
        key_points = self._coerce_string_list(payload.get("key_points"))[:3]
        topic_tags = self._coerce_string_list(payload.get("topic_tags"))[:3]
        notes = [f"Chunk {index} summary: {summary}"]
        if key_points:
            notes.append("Key points: " + "; ".join(key_points))
        if topic_tags:
            notes.append("Topic tags: " + ", ".join(topic_tags))
        return "\n".join(notes)

    def _coerce_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [self._coerce_text(item) for item in value if self._coerce_text(item)]
        if isinstance(value, str) and value.strip():
            return [part.strip("- ").strip() for part in value.split("\n") if part.strip()]
        return []

    def _coerce_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(value)
        return output
