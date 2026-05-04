from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

from app.core.models import ConsensusResult, HandoffState, WorkerResult


class ConsensusAgent:
    async def combine(
        self,
        handoff: HandoffState,
        db_result: WorkerResult,
        vector_result: WorkerResult,
    ) -> tuple[ConsensusResult, int]:
        started = perf_counter()
        winning_sources: list[str] = []
        notes: list[str] = []
        fragments: list[str] = []

        if db_result.success and db_result.payload.get("found"):
            winning_sources.append("db")
            record = db_result.payload["record"]
            fragments.append(
                "Account snapshot: "
                f"plan={record['plan']}; billing_status={record['billing_status']}; "
                f"invoice_id={record['invoice_id']}; amount_usd={record['invoice_amount_usd']}; "
                f"next_bill_date={record['next_bill_date']}; payment_method={record['payment_method']}; "
                f"balance_due_usd={record['balance_due_usd']}."
            )
        elif not db_result.success:
            notes.append(f"DB agent failed: {db_result.error}")
        else:
            notes.append("DB agent found no matching account record.")

        docs = vector_result.payload.get("documents", []) if vector_result.success else []
        if docs:
            winning_sources.append("vector")
            top_doc = docs[0]
            fragments.append(f"Policy guidance: {top_doc['body']}")
        elif not vector_result.success:
            notes.append(f"Vector agent failed: {vector_result.error}")
        else:
            notes.append("Vector agent found no matching policy documents.")

        fallback_used = not winning_sources
        if fallback_used:
            fragments.append(
                "Fallback guidance: apologize briefly, confirm the request, and offer a manual review path."
            )

        confidence = min(0.98, max(db_result.confidence, vector_result.confidence))
        if db_result.success and vector_result.success and winning_sources == ["db", "vector"]:
            confidence = max(confidence, 0.9)

        if db_result.success and vector_result.success and docs:
            status = db_result.payload.get("record", {}).get("billing_status")
            if status == "past_due" and "refund" in handoff.current_intent.lower():
                notes.append(
                    "Past-due status may limit immediate refund actions until the account is verified."
                )

        latency_ms = int((perf_counter() - started) * 1000)
        return (
            ConsensusResult(
                answer_context=" ".join(fragments),
                confidence=confidence,
                winning_sources=winning_sources,
                conflict_notes=notes,
                fallback_used=fallback_used,
            ),
            latency_ms,
        )


class AsyncBarrier:
    async def gather(self, *coroutines: Any) -> list[Any]:
        return list(await asyncio.gather(*coroutines, return_exceptions=True))
