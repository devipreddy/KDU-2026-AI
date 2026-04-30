"""Cost control utilities for model routing, context budgeting, and caching."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import to_plain_data
from ..settings import AppSettings


def _stable_json(value: Any) -> str:
    return json.dumps(to_plain_data(value), sort_keys=True, separators=(",", ":"))


def stable_cache_key(namespace: str, value: Any) -> str:
    import hashlib

    payload = f"{namespace}:{_stable_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class ModelRouteTable:
    extraction: str
    planning: str
    execution: str
    summarization: str
    coordinator: str
    finance: str
    hr: str


class ModelRouter:
    def __init__(self, routes: ModelRouteTable) -> None:
        self.routes = routes

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "ModelRouter":
        return cls(
            ModelRouteTable(
                extraction=settings.extraction_model,
                planning=settings.planner_model,
                execution=settings.executor_model,
                summarization=settings.summarization_model,
                coordinator=settings.coordinator_model,
                finance=settings.finance_model,
                hr=settings.hr_model,
            )
        )

    def route_for(self, task_name: str) -> str:
        if not hasattr(self.routes, task_name):
            raise KeyError(f"Unsupported model routing task: {task_name}")
        return getattr(self.routes, task_name)

    def as_dict(self) -> dict[str, str]:
        return {
            "extraction": self.routes.extraction,
            "planning": self.routes.planning,
            "execution": self.routes.execution,
            "summarization": self.routes.summarization,
            "coordinator": self.routes.coordinator,
            "finance": self.routes.finance,
            "hr": self.routes.hr,
        }


@dataclass(slots=True)
class ContextBudgetReport:
    max_tokens: int
    original_estimated_tokens: int
    final_estimated_tokens: int
    compressed: bool
    removed_short_term_messages: int = 0
    trimmed_case_message_count: int = 0
    removed_raw_excerpts: int = 0
    notes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "original_estimated_tokens": self.original_estimated_tokens,
            "final_estimated_tokens": self.final_estimated_tokens,
            "compressed": self.compressed,
            "removed_short_term_messages": self.removed_short_term_messages,
            "trimmed_case_message_count": self.trimmed_case_message_count,
            "removed_raw_excerpts": self.removed_raw_excerpts,
            "notes": list(self.notes or []),
        }


class TokenBudgetManager:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def estimate_tokens(self, value: Any) -> int:
        if value is None:
            return 0
        serialized = _stable_json(value)
        if not serialized:
            return 0
        return max(1, math.ceil(len(serialized) / 4))

    def compress_context(self, payload: dict[str, Any]) -> tuple[dict[str, Any], ContextBudgetReport]:
        compressed = json.loads(json.dumps(to_plain_data(payload)))
        notes: list[str] = []
        original_tokens = self.estimate_tokens(compressed)
        report = ContextBudgetReport(
            max_tokens=self.max_tokens,
            original_estimated_tokens=original_tokens,
            final_estimated_tokens=original_tokens,
            compressed=False,
            notes=notes,
        )
        if original_tokens <= self.max_tokens:
            return compressed, report

        report.compressed = True

        short_term_window = compressed.get("short_term_window")
        if isinstance(short_term_window, list):
            while len(short_term_window) > 3 and self.estimate_tokens(compressed) > self.max_tokens:
                short_term_window.pop(0)
                report.removed_short_term_messages += 1
            for index, message in enumerate(list(short_term_window)):
                if self.estimate_tokens(compressed) <= self.max_tokens:
                    break
                if isinstance(message, str) and len(message) > 160:
                    short_term_window[index] = message[:157] + "..."
                    notes.append("Trimmed long short-term messages to stay within token budget.")

        case_facts = compressed.get("case_facts")
        if isinstance(case_facts, dict):
            recent_messages = case_facts.get("recent_relevant_messages")
            if isinstance(recent_messages, list):
                while len(recent_messages) > 3 and self.estimate_tokens(compressed) > self.max_tokens:
                    recent_messages.pop(0)
                    report.trimmed_case_message_count += 1
            for collection_name in ("orders", "transactions"):
                collection = case_facts.get(collection_name)
                if isinstance(collection, dict):
                    for item in collection.values():
                        if (
                            isinstance(item, dict)
                            and "raw_excerpt" in item
                            and item["raw_excerpt"] is not None
                        ):
                            item["raw_excerpt"] = None
                            report.removed_raw_excerpts += 1
                    if report.removed_raw_excerpts:
                        notes.append("Dropped verbose raw excerpts while preserving IDs and amounts.")

        if self.estimate_tokens(compressed) > self.max_tokens:
            working_memory = compressed.get("working_memory")
            if isinstance(working_memory, dict) and "recent_decisions" in working_memory:
                decisions = working_memory.get("recent_decisions")
                if isinstance(decisions, list) and len(decisions) > 1:
                    working_memory["recent_decisions"] = decisions[-1:]
                    notes.append("Kept only the most recent decision in working memory.")

        report.final_estimated_tokens = self.estimate_tokens(compressed)
        return compressed, report


@dataclass(slots=True)
class CacheLookup:
    hit: bool
    key: str
    value: Any = None


class JsonFileCache:
    def __init__(self, base_dir: Path, namespace: str, ttl_seconds: int) -> None:
        self.base_dir = base_dir / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.base_dir / f"{key}.json"

    def get(self, key: str) -> CacheLookup:
        path = self._path(key)
        if not path.exists():
            return CacheLookup(hit=False, key=key)

        payload = json.loads(path.read_text(encoding="utf-8"))
        created_at = payload.get("created_at", 0)
        if self.ttl_seconds > 0 and time.time() - created_at > self.ttl_seconds:
            path.unlink(missing_ok=True)
            return CacheLookup(hit=False, key=key)

        return CacheLookup(hit=True, key=key, value=payload.get("value"))

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        payload = {
            "created_at": time.time(),
            "value": to_plain_data(value),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
