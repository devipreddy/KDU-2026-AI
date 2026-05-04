from __future__ import annotations

from time import perf_counter

from app.core.models import HandoffState, WorkerResult
from app.storage.repositories import AccountRepository


class DBAgent:
    name = "db"

    def __init__(self, repository: AccountRepository) -> None:
        self.repository = repository

    async def run(self, handoff: HandoffState) -> WorkerResult:
        started = perf_counter()
        try:
            payload = await self.repository.lookup(handoff.user_id, handoff.user_query)
            latency_ms = int((perf_counter() - started) * 1000)
            confidence = 0.94 if payload.get("found") else 0.2
            return WorkerResult(
                agent="db",
                success=True,
                latency_ms=latency_ms,
                confidence=confidence,
                payload=payload,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            latency_ms = int((perf_counter() - started) * 1000)
            return WorkerResult(
                agent="db",
                success=False,
                latency_ms=latency_ms,
                confidence=0.0,
                error=str(exc),
            )
