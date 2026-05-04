from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


class QueueOverloadedError(RuntimeError):
    pass


@dataclass(slots=True)
class QueueSnapshot:
    limit: int
    backlog: int
    queued: int
    in_flight: int


class ConcurrencyQueue:
    def __init__(self, limit: int, backlog: int) -> None:
        self._limit = limit
        self._backlog = backlog
        self._queued = 0
        self._in_flight = 0
        self._guard = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(limit)

    async def run(self, operation_name: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        async with self._guard:
            if self._queued >= self._backlog:
                raise QueueOverloadedError(f"{operation_name} backlog exceeded")
            self._queued += 1

        started = perf_counter()
        try:
            async with self._semaphore:
                async with self._guard:
                    self._queued -= 1
                    self._in_flight += 1
                result = await factory()
                latency_ms = int((perf_counter() - started) * 1000)
                return result, latency_ms
        finally:
            async with self._guard:
                if self._in_flight > 0:
                    self._in_flight -= 1

    async def snapshot(self) -> QueueSnapshot:
        async with self._guard:
            return QueueSnapshot(
                limit=self._limit,
                backlog=self._backlog,
                queued=self._queued,
                in_flight=self._in_flight,
            )
