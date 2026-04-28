from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ApiUsageEntry:
    operation: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    response_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedSection:
    text: str
    page_number: int | None = None
    source: str = "unknown"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    sections: list[ExtractedSection]
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage_entries: list[ApiUsageEntry] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedTextResult:
    sections: list[ExtractedSection]
    extracted_text: str
    cleaned_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnrichmentResult:
    title: str | None
    summary: str
    key_points: list[str]
    topic_tags: list[str]
    usage_entries: list[ApiUsageEntry] = field(default_factory=list)


@dataclass(slots=True)
class ChunkPayload:
    chunk_id: str
    file_id: str
    chunk_index: int
    content: str
    token_count: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisionExtractionResult:
    extracted_text: str
    visual_description: str
    layout_notes: str
    detected_language: str
    usage_entry: ApiUsageEntry


@dataclass(slots=True)
class VisualDescriptionResult:
    visual_description: str
    layout_notes: str
    contains_meaningful_visuals: bool
    usage_entry: ApiUsageEntry
