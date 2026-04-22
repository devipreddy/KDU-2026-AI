from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Awaitable, Callable, Deque, Literal


class CircuitBreakerOpenError(RuntimeError):
    pass


CircuitState = Literal["closed", "open", "half_open"]


@dataclass
class CircuitBreakerSnapshot:
    name: str
    state: CircuitState
    failure_rate: float
    sample_size: int
    last_error: str | None
    recovery_timeout_seconds: int
    retry_after_seconds: int


class AsyncCircuitBreaker:
    def __init__(
        self,
        *,
        name: str,
        window_size: int = 6,
        failure_rate_threshold: float = 0.5,
        minimum_calls: int = 4,
        recovery_timeout_seconds: int = 45,
    ) -> None:
        self.name = name
        self.window_size = max(2, window_size)
        self.failure_rate_threshold = min(max(failure_rate_threshold, 0.0), 1.0)
        self.minimum_calls = max(1, minimum_calls)
        self.recovery_timeout_seconds = max(1, recovery_timeout_seconds)
        self._state: CircuitState = "closed"
        self._opened_at: float | None = None
        self._last_error: str | None = None
        self._probe_in_flight = False
        self._outcomes: Deque[bool] = deque(maxlen=self.window_size)
        self._lock = RLock()

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            self._after_failure(str(exc))
            raise
        self._after_success()
        return result

    def snapshot(self) -> CircuitBreakerSnapshot:
        with self._lock:
            state = self._current_state_locked()
            failure_rate = self._failure_rate_locked()
            retry_after_seconds = self._retry_after_seconds_locked() if state == "open" else 0
            return CircuitBreakerSnapshot(
                name=self.name,
                state=state,
                failure_rate=round(failure_rate, 3),
                sample_size=len(self._outcomes),
                last_error=self._last_error,
                recovery_timeout_seconds=self.recovery_timeout_seconds,
                retry_after_seconds=retry_after_seconds,
            )

    def _before_call(self) -> None:
        with self._lock:
            state = self._current_state_locked()
            if state == "open":
                retry_after = self._retry_after_seconds_locked()
                raise CircuitBreakerOpenError(
                    f"{self.name} circuit breaker is open; retry in about {retry_after}s."
                )
            if state == "half_open":
                if self._probe_in_flight:
                    raise CircuitBreakerOpenError(
                        f"{self.name} circuit breaker is half-open and probing recovery."
                    )
                self._probe_in_flight = True

    def _after_success(self) -> None:
        with self._lock:
            self._outcomes.append(True)
            self._last_error = None
            self._probe_in_flight = False
            self._state = "closed"
            self._opened_at = None

    def _after_failure(self, message: str) -> None:
        with self._lock:
            self._outcomes.append(False)
            self._last_error = message
            self._probe_in_flight = False
            if self._state == "half_open":
                self._trip_locked()
                return
            if len(self._outcomes) >= self.minimum_calls and (
                self._failure_rate_locked() >= self.failure_rate_threshold
            ):
                self._trip_locked()

    def _trip_locked(self) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()

    def _current_state_locked(self) -> CircuitState:
        if self._state == "open" and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout_seconds:
                self._state = "half_open"
                self._probe_in_flight = False
        return self._state

    def _retry_after_seconds_locked(self) -> int:
        if self._opened_at is None:
            return 0
        remaining = self.recovery_timeout_seconds - (time.monotonic() - self._opened_at)
        return max(0, int(remaining))

    def _failure_rate_locked(self) -> float:
        if not self._outcomes:
            return 0.0
        failures = sum(1 for outcome in self._outcomes if not outcome)
        return failures / len(self._outcomes)
