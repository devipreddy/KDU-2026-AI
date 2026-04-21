from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SummaryLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @classmethod
    def from_value(cls, value: str | "SummaryLength") -> "SummaryLength":
        if isinstance(value, cls):
            return value

        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"Unsupported summary length: {value}")


@dataclass(frozen=True)
class LengthProfile:
    label: SummaryLength
    min_words: int
    max_words: int
    instructions: str


@dataclass(frozen=True)
class ModelSettings:
    model_name: str
    task: str
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    max_input_tokens: int = 768
    chunk_overlap_tokens: int = 64


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    qa_confidence_threshold: float = 0.2


@dataclass(frozen=True)
class AssistantSettings:
    runtime: RuntimeSettings
    length_profiles: dict[SummaryLength, LengthProfile]
    summarizer: ModelSettings
    refiner: ModelSettings
    qa: ModelSettings


@dataclass(frozen=True)
class SummaryArtifacts:
    source_text: str
    chunk_count: int
    draft_summary: str
    final_summary: str
    length: SummaryLength


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    score: float
    is_grounded: bool
