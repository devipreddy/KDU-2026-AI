"""Tracing, metrics, and alerting for workflow execution."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ObservabilityManager:
    def __init__(self, base_dir: Path, workflow_name: str, session_id: str) -> None:
        self.base_dir = base_dir / "observability"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_name = workflow_name
        self.session_id = session_id
        self.traces_path = self.base_dir / f"{workflow_name}-{session_id}-traces.jsonl"
        self.alerts_path = self.base_dir / f"{workflow_name}-{session_id}-alerts.jsonl"
        self.metrics_path = self.base_dir / f"{workflow_name}-{session_id}-metrics.json"
        self._metrics = {
            "trace_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "response_cache_hits": 0,
            "tool_cache_hits": 0,
            "alerts_total": 0,
            "latency_total_ms": 0.0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._flush_metrics()

    def trace(
        self,
        *,
        agent: str,
        tool: str,
        latency_ms: float,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "created_at": time.time(),
            "workflow": self.workflow_name,
            "session_id": self.session_id,
            "agent": agent,
            "tool": tool,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "details": details or {},
        }
        with self.traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

        self._metrics["trace_count"] += 1
        self._metrics["latency_total_ms"] += latency_ms
        if status not in {"success", "completed", "cache_hit"}:
            self._metrics["failure_count"] += 1
        self._flush_metrics()
        LOGGER.info(json.dumps(event, sort_keys=True))

    def record_tokens(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        token_usage = self._metrics["token_usage"]
        token_usage["prompt_tokens"] += prompt_tokens
        token_usage["completion_tokens"] += completion_tokens
        token_usage["total_tokens"] += prompt_tokens + completion_tokens
        self._flush_metrics()

    def record_retry(self, count: int = 1) -> None:
        self._metrics["retry_count"] += count
        self._flush_metrics()

    def record_response_cache_hit(self) -> None:
        self._metrics["response_cache_hits"] += 1
        self._flush_metrics()

    def record_tool_cache_hit(self) -> None:
        self._metrics["tool_cache_hits"] += 1
        self._flush_metrics()

    def alert(
        self,
        *,
        alert_type: str,
        message: str,
        severity: str = "warning",
        context: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "created_at": time.time(),
            "workflow": self.workflow_name,
            "session_id": self.session_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "context": context or {},
        }
        with self.alerts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        self._metrics["alerts_total"] += 1
        self._flush_metrics()
        LOGGER.warning(json.dumps(event, sort_keys=True))

    def snapshot(self) -> dict[str, Any]:
        trace_count = self._metrics["trace_count"]
        failure_count = self._metrics["failure_count"]
        latency_total_ms = self._metrics["latency_total_ms"]
        return {
            "workflow": self.workflow_name,
            "session_id": self.session_id,
            "trace_count": trace_count,
            "failure_count": failure_count,
            "retry_count": self._metrics["retry_count"],
            "failure_rate": (failure_count / trace_count) if trace_count else 0.0,
            "avg_latency_ms": (latency_total_ms / trace_count) if trace_count else 0.0,
            "response_cache_hits": self._metrics["response_cache_hits"],
            "tool_cache_hits": self._metrics["tool_cache_hits"],
            "alerts_total": self._metrics["alerts_total"],
            "token_usage": dict(self._metrics["token_usage"]),
        }

    def _flush_metrics(self) -> None:
        self.metrics_path.write_text(
            json.dumps(self.snapshot(), indent=2),
            encoding="utf-8",
        )
