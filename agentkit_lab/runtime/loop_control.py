"""Loop tracking and failure fingerprinting utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..schemas import LoopDetectionEvent
from .circuit_breaker import CircuitBreaker
from .retry_policy import FailureClassification, RetryBudget


def fingerprint_arguments(tool_name: str, arguments: str) -> str:
    raw = f"{tool_name}:{arguments}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


@dataclass(slots=True)
class ToolFailureTracker:
    circuit_breaker: CircuitBreaker
    retry_budget: RetryBudget
    total_attempts: dict[str, int] = field(default_factory=dict)
    loop_events: list[LoopDetectionEvent] = field(default_factory=list)
    latest_feedback: dict[str, object] | None = None

    def record_attempt(self, tool_name: str) -> None:
        self.total_attempts[tool_name] = self.total_attempts.get(tool_name, 0) + 1

    def attempts_for(self, tool_name: str) -> int:
        return self.total_attempts.get(tool_name, 0)

    def record_failure(
        self,
        tool_name: str,
        arguments: str,
        classification: FailureClassification,
        error_message: str,
    ) -> LoopDetectionEvent | None:
        state = self.circuit_breaker.record_failure(tool_name, error_message)
        self.latest_feedback = {
            "tool_name": tool_name,
            "tool_error": error_message,
            "error_type": classification.error_type,
            "retryable": classification.retryable,
            "instruction": "Evaluate retry policy before calling the tool again.",
            "retry_budget_remaining": self.retry_budget.remaining,
            "circuit_breaker_open": state.is_open,
        }
        if state.is_open:
            event = LoopDetectionEvent(
                tool_name=tool_name,
                consecutive_failures=state.consecutive_failures,
                argument_fingerprint=fingerprint_arguments(tool_name, arguments),
                reason=error_message,
            )
            self.loop_events.append(event)
            return event
        return None

    def record_success(self, tool_name: str) -> None:
        self.circuit_breaker.record_success(tool_name)
        self.latest_feedback = None

    def build_feedback(
        self,
        *,
        tool_name: str,
        classification: FailureClassification,
        error_message: str,
        retry_allowed: bool,
        backoff_seconds: float,
        instruction: str,
    ) -> dict[str, object]:
        feedback = {
            "tool_name": tool_name,
            "tool_error": error_message,
            "error_type": classification.error_type,
            "retryable": classification.retryable,
            "retry_allowed": retry_allowed,
            "backoff_seconds": backoff_seconds,
            "instruction": instruction,
            "retry_budget_remaining": self.retry_budget.remaining,
            "circuit_breaker_open": not self.circuit_breaker.allow(tool_name),
        }
        self.latest_feedback = feedback
        return feedback
