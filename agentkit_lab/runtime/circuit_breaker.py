"""Circuit breaker implementation for repeated tool failures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CircuitState:
    consecutive_failures: int = 0
    is_open: bool = False
    last_error: str | None = None


@dataclass(slots=True)
class CircuitBreaker:
    threshold: int
    _state: dict[str, CircuitState] = field(default_factory=dict)

    def state_for(self, key: str) -> CircuitState:
        return self._state.setdefault(key, CircuitState())

    def allow(self, key: str) -> bool:
        return not self.state_for(key).is_open

    def record_success(self, key: str) -> None:
        state = self.state_for(key)
        state.consecutive_failures = 0
        state.is_open = False
        state.last_error = None

    def record_failure(self, key: str, error_message: str) -> CircuitState:
        state = self.state_for(key)
        state.consecutive_failures += 1
        state.last_error = error_message
        if state.consecutive_failures >= self.threshold:
            state.is_open = True
        return state
