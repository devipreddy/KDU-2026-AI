from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


class AccountRepository:
    def __init__(self, path: Path) -> None:
        self._records = json.loads(path.read_text(encoding="utf-8"))

    async def lookup(self, user_id: str, transcript: str) -> dict[str, Any]:
        await asyncio.sleep(0.15)
        lowered = transcript.lower()
        record = next((item for item in self._records if item["user_id"] == user_id), None)
        if record is None:
            record = next(
                (
                    item
                    for item in self._records
                    if item["invoice_id"].lower() in lowered
                    or item["payment_method"].split()[-1].lower() in lowered
                ),
                None,
            )
        if record is None:
            return {
                "found": False,
                "reason": "No matching billing account found in the mock database.",
            }
        return {
            "found": True,
            "record": record,
            "reason": "Matched by user id or billing clue.",
        }


class VectorRepository:
    def __init__(self, path: Path) -> None:
        self._docs = json.loads(path.read_text(encoding="utf-8"))

    async def search(self, query: str, limit: int = 2) -> dict[str, Any]:
        await asyncio.sleep(0.22)
        query_terms = {term for term in query.lower().replace("-", " ").split() if len(term) > 2}
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self._docs:
            haystack = " ".join([doc["title"], doc["body"], " ".join(doc.get("tags", []))]).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [doc for _, doc in scored[:limit]]
        return {
            "found": bool(top),
            "documents": top,
        }
