from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from threading import RLock


@dataclass
class MetricsSnapshot:
    total_requests: int = 0
    total_latency_ms: int = 0
    llm_calls: int = 0
    tool_calls: Counter[str] = field(default_factory=Counter)
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class MetricsStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._data = MetricsSnapshot()

    def record_request(
        self,
        *,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool,
        llm_calls: int,
        tool_names: list[str],
    ) -> None:
        with self._lock:
            self._data.total_requests += 1
            self._data.total_latency_ms += latency_ms
            self._data.total_input_tokens += input_tokens
            self._data.total_output_tokens += output_tokens
            self._data.llm_calls += llm_calls
            if cache_hit:
                self._data.cache_hits += 1
            self._data.tool_calls.update(tool_names)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            total_requests = self._data.total_requests
            avg_latency = (
                self._data.total_latency_ms / total_requests if total_requests else 0.0
            )
            total_tokens = self._data.total_input_tokens + self._data.total_output_tokens
            cache_hit_rate = self._data.cache_hits / total_requests if total_requests else 0.0
            return {
                "total_requests": total_requests,
                "avg_latency_ms": round(avg_latency, 2),
                "llm_calls": self._data.llm_calls,
                "tool_calls": dict(self._data.tool_calls),
                "cache_hit_rate": round(cache_hit_rate, 4),
                "token_usage": {
                    "input": self._data.total_input_tokens,
                    "output": self._data.total_output_tokens,
                    "total": total_tokens,
                },
            }
