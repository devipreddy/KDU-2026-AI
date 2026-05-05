from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class RealtimeHub:
    def __init__(self) -> None:
        self._thread_subscribers: dict[str, set[WebSocket]] = defaultdict(set)
        self._queue_subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect_thread(self, thread_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._thread_subscribers[thread_id].add(websocket)

    async def connect_queue(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._queue_subscribers.add(websocket)

    async def disconnect_thread(self, thread_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            subscribers = self._thread_subscribers.get(thread_id)
            if not subscribers:
                return
            subscribers.discard(websocket)
            if not subscribers:
                self._thread_subscribers.pop(thread_id, None)

    async def disconnect_queue(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._queue_subscribers.discard(websocket)

    async def broadcast_thread(
        self,
        thread_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._broadcast(
            list(self._thread_subscribers.get(thread_id, set())),
            {
                "type": event_type,
                "thread_id": thread_id,
                "payload": payload or {},
            },
        )

    async def broadcast_queue(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._broadcast(
            list(self._queue_subscribers),
            {
                "type": event_type,
                "payload": payload or {},
            },
        )

    async def _broadcast(
        self,
        sockets: list[WebSocket],
        message: dict[str, Any],
    ) -> None:
        stale: list[WebSocket] = []
        for websocket in sockets:
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._queue_subscribers.discard(websocket)
                    for subscribers in self._thread_subscribers.values():
                        subscribers.discard(websocket)

