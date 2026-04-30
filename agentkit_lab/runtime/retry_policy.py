"""Retry budgeting, failure classification, and backoff policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..exceptions import (
    InternalDatabaseAuthError,
    InternalDatabaseBadRequestError,
    InternalDatabaseTimeoutError,
    InternalDatabaseUnavailableError,
)


@dataclass(slots=True)
class RetryBudget:
    """Request-level retry budget shared across all tools."""

    max_attempts: int = 5
    remaining: int = field(init=False)
    consumed: int = 0

    def __post_init__(self) -> None:
        self.remaining = self.max_attempts

    def consume(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.consumed += 1
        return True


@dataclass(slots=True)
class FailureClassification:
    error_type: str
    retryable: bool
    reason: str
    max_retry_count: int | None = None


@dataclass(slots=True)
class ExponentialBackoffPolicy:
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def delay_for(self, retry_count: int) -> float:
        if retry_count <= 0:
            return 0.0
        delay = self.base_delay_seconds * (2 ** (retry_count - 1))
        return min(delay, self.max_delay_seconds)


def classify_failure(error: Exception) -> FailureClassification:
    if isinstance(error, InternalDatabaseUnavailableError):
        return FailureClassification(
            error_type="server_error",
            retryable=True,
            reason="Temporary internal server error.",
        )
    if isinstance(error, InternalDatabaseTimeoutError):
        return FailureClassification(
            error_type="timeout",
            retryable=True,
            reason="Temporary timeout while contacting the service.",
            max_retry_count=2,
        )
    if isinstance(error, InternalDatabaseBadRequestError):
        return FailureClassification(
            error_type="bad_request",
            retryable=False,
            reason="The request to the service is invalid.",
        )
    if isinstance(error, InternalDatabaseAuthError):
        return FailureClassification(
            error_type="auth_error",
            retryable=False,
            reason="The service rejected authentication.",
        )
    return FailureClassification(
        error_type="unknown_error",
        retryable=False,
        reason="Unknown service failure.",
    )
