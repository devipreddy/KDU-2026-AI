from __future__ import annotations

import re
import uuid
from typing import Iterable

import tiktoken

from app.services.types import ChunkPayload, ExtractedSection


class _SimpleEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"\s+|\w+|[^\w\s]", text, re.UNICODE)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


class TokenChunker:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = _SimpleEncoding()

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        file_id: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[ChunkPayload]:
        units = self._build_units(sections, max_tokens)
        if not units:
            return []

        chunks: list[ChunkPayload] = []
        current_units: list[dict[str, object]] = []
        current_tokens = 0

        for unit in units:
            unit_text = str(unit["text"])
            unit_tokens = int(unit["token_count"])
            if current_units and current_tokens + unit_tokens > max_tokens:
                chunks.append(self._finalize_chunk(file_id, len(chunks), current_units))
                current_units, current_tokens = self._make_overlap(current_units, overlap_tokens)

            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            chunks.append(self._finalize_chunk(file_id, len(chunks), current_units))

        return chunks

    def _build_units(self, sections: list[ExtractedSection], max_tokens: int) -> list[dict[str, object]]:
        units: list[dict[str, object]] = []
        for section in sections:
            raw_paragraphs = [part.strip() for part in re.split(r"\n{2,}", section.text) if part.strip()]
            if not raw_paragraphs and section.text.strip():
                raw_paragraphs = [section.text.strip()]

            for paragraph in raw_paragraphs:
                units.extend(self._split_text_unit(paragraph, section, max_tokens))
        return units

    def _split_text_unit(
        self,
        text: str,
        section: ExtractedSection,
        max_tokens: int,
    ) -> list[dict[str, object]]:
        token_count = self.count_tokens(text)
        if token_count <= max_tokens:
            return [self._make_unit(text, token_count, section)]

        parts: list[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and self.count_tokens(candidate) > max_tokens:
                parts.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current.strip():
            parts.append(current.strip())

        output: list[dict[str, object]] = []
        for part in parts:
            part_tokens = self.count_tokens(part)
            if part_tokens <= max_tokens:
                output.append(self._make_unit(part, part_tokens, section))
                continue

            token_ids = self.encoding.encode(part)
            for start in range(0, len(token_ids), max_tokens):
                token_window = token_ids[start : start + max_tokens]
                window_text = self.encoding.decode(token_window).strip()
                if window_text:
                    output.append(self._make_unit(window_text, len(token_window), section))
        return output

    def _make_unit(self, text: str, token_count: int, section: ExtractedSection) -> dict[str, object]:
        return {
            "text": text,
            "token_count": token_count,
            "page_number": section.page_number,
            "source": section.source,
        }

    def _finalize_chunk(
        self,
        file_id: str,
        chunk_index: int,
        units: list[dict[str, object]],
    ) -> ChunkPayload:
        content = "\n\n".join(str(unit["text"]) for unit in units).strip()
        page_numbers = sorted(
            {
                int(page_number)
                for page_number in (unit["page_number"] for unit in units)
                if page_number is not None
            }
        )
        sources = sorted({str(unit["source"]) for unit in units})
        return ChunkPayload(
            chunk_id=uuid.uuid4().hex,
            file_id=file_id,
            chunk_index=chunk_index,
            content=content,
            token_count=self.count_tokens(content),
            page_number=page_numbers[0] if page_numbers else None,
            metadata={"page_numbers": page_numbers, "sources": sources},
        )

    def _make_overlap(
        self,
        current_units: list[dict[str, object]],
        overlap_tokens: int,
    ) -> tuple[list[dict[str, object]], int]:
        if overlap_tokens <= 0:
            return [], 0

        overlap: list[dict[str, object]] = []
        running_tokens = 0
        for unit in reversed(current_units):
            unit_tokens = int(unit["token_count"])
            if not overlap and unit_tokens > overlap_tokens:
                token_ids = self.encoding.encode(str(unit["text"]))
                overlap_slice = token_ids[-overlap_tokens:]
                overlap.insert(
                    0,
                    {
                        **unit,
                        "text": self.encoding.decode(overlap_slice).strip(),
                        "token_count": len(overlap_slice),
                    },
                )
                running_tokens = len(overlap_slice)
                break
            overlap.insert(0, unit)
            running_tokens += unit_tokens
            if running_tokens >= overlap_tokens:
                break
        return overlap, running_tokens

    def trim_text(self, text: str, max_tokens: int) -> str:
        token_ids = self.encoding.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        return self.encoding.decode(token_ids[:max_tokens]).strip()

    def split_text(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        section = ExtractedSection(text=text, source="text")
        return [
            chunk.content
            for chunk in self.chunk_sections([section], "ephemeral", max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        ]
