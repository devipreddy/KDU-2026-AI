from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.agents.billing import BillingAgent
from app.agents.consensus import ConsensusAgent
from app.agents.db_agent import DBAgent
from app.agents.triage import TriageAgent
from app.agents.vector_agent import VectorAgent
from app.core.models import (
    AgentObservation,
    ConversationMessage,
    HandoffState,
    SessionPhase,
    SessionSnapshot,
    TriageDecision,
    WorkerResult,
)
from app.services.concurrency import ConcurrencyQueue
from app.services.pruning import ConversationPruner
from app.services.security import SecurityGuard


class Coordinator:
    def __init__(
        self,
        *,
        triage_agent: TriageAgent,
        billing_agent: BillingAgent,
        db_agent: DBAgent,
        vector_agent: VectorAgent,
        consensus_agent: ConsensusAgent,
        pruner: ConversationPruner,
        security: SecurityGuard,
        db_queue: ConcurrencyQueue,
        vector_queue: ConcurrencyQueue,
        worker_timeout_seconds: float,
    ) -> None:
        self.triage_agent = triage_agent
        self.billing_agent = billing_agent
        self.db_agent = db_agent
        self.vector_agent = vector_agent
        self.consensus_agent = consensus_agent
        self.pruner = pruner
        self.security = security
        self.db_queue = db_queue
        self.vector_queue = vector_queue
        self.worker_timeout_seconds = worker_timeout_seconds

    async def handle_turn(self, snapshot: SessionSnapshot, transcript: str) -> dict[str, Any]:
        clean_transcript = self.security.validate_user_text(transcript)
        snapshot.phase = SessionPhase.PROCESSING
        snapshot.metrics.turns += 1
        snapshot.last_user_transcript = clean_transcript
        snapshot.recent_messages.append(ConversationMessage(role="user", text=clean_transcript, agent="user"))
        snapshot.conversation_summary, snapshot.recent_messages = self.pruner.prune(
            snapshot.conversation_summary,
            snapshot.recent_messages,
        )

        observations: list[AgentObservation] = []
        triage_started = perf_counter()
        triage_decision, triage_usage = await self.triage_agent.route(clean_transcript, snapshot)
        triage_latency = int((perf_counter() - triage_started) * 1000)
        observations.append(
            AgentObservation(
                agent="triage",
                action="route",
                detail=triage_decision.routing_reason,
                latency_ms=triage_latency,
                metadata={"usage": triage_usage or {}},
            )
        )

        handoff = self._build_handoff(snapshot, triage_decision, clean_transcript)
        snapshot.pending_handoff = handoff

        if triage_decision.target_agent == "triage":
            reply_text = (
                "I can help route that request. Right now this demo is optimized for billing flows, "
                "so please ask a billing question like invoice, refund, payment method, or duplicate charge."
            )
            snapshot.last_assistant_reply = reply_text
            return {
                "snapshot": snapshot,
                "handoff": handoff,
                "observations": observations,
                "worker_results": [],
                "consensus": None,
                "assistant_text": reply_text,
            }

        db_task = asyncio.create_task(
            self.db_queue.run("db_lookup", lambda: self.db_agent.run(handoff))
        )
        vector_task = asyncio.create_task(
            self.vector_queue.run("vector_search", lambda: self.vector_agent.run(handoff))
        )
        db_outcome, vector_outcome = await asyncio.gather(
            asyncio.wait_for(db_task, timeout=self.worker_timeout_seconds),
            asyncio.wait_for(vector_task, timeout=self.worker_timeout_seconds),
            return_exceptions=True,
        )
        worker_results = [
            self._coerce_worker_result("db", db_outcome),
            self._coerce_worker_result("vector", vector_outcome),
        ]
        for result in worker_results:
            observations.append(
                AgentObservation(
                    agent=result.agent,
                    action="lookup",
                    detail="success" if result.success else (result.error or "failed"),
                    latency_ms=result.latency_ms,
                )
            )

        consensus, consensus_latency = await self.consensus_agent.combine(
            handoff,
            worker_results[0],
            worker_results[1],
        )
        observations.append(
            AgentObservation(
                agent="consensus",
                action="merge",
                detail=", ".join(consensus.winning_sources) or "fallback",
                latency_ms=consensus_latency,
            )
        )

        billing_started = perf_counter()
        billing_reply, billing_usage = await self.billing_agent.respond(handoff, consensus)
        billing_latency = int((perf_counter() - billing_started) * 1000)
        observations.append(
            AgentObservation(
                agent="billing",
                action="respond",
                detail=billing_reply.short_summary,
                latency_ms=billing_latency,
                metadata={"usage": billing_usage or {}},
            )
        )

        snapshot.last_assistant_reply = billing_reply.reply_text
        snapshot.current_agent = "billing"

        return {
            "snapshot": snapshot,
            "handoff": handoff,
            "observations": observations,
            "worker_results": worker_results,
            "consensus": consensus,
            "assistant_text": billing_reply.reply_text,
        }

    def _build_handoff(
        self,
        snapshot: SessionSnapshot,
        decision: TriageDecision,
        transcript: str,
    ) -> HandoffState:
        return HandoffState(
            trace_id=snapshot.trace_id,
            session_id=snapshot.session_id,
            turn_id=f"turn_{uuid4().hex}",
            user_id=snapshot.user_id,
            source_agent="triage",
            target_agent=decision.target_agent,
            current_intent=decision.intent,
            intent_confidence=decision.confidence,
            entities=decision.entities,
            user_query=transcript,
            conversation_summary=snapshot.conversation_summary,
            recent_messages=list(snapshot.recent_messages),
            metadata={
                "user_goal": decision.user_goal,
                "routing_reason": decision.routing_reason,
                "policy_flags": decision.policy_flags,
            },
        )

    def _coerce_worker_result(self, agent: str, outcome: Any) -> WorkerResult:
        if isinstance(outcome, asyncio.TimeoutError):
            return WorkerResult(
                agent=agent,
                success=False,
                latency_ms=int(self.worker_timeout_seconds * 1000),
                confidence=0.0,
                error=f"{agent} worker timed out after {self.worker_timeout_seconds:.1f}s",
            )
        if isinstance(outcome, Exception):
            return WorkerResult(
                agent=agent,
                success=False,
                latency_ms=0,
                confidence=0.0,
                error=str(outcome),
            )
        if isinstance(outcome, tuple):
            result, latency_ms = outcome
            if isinstance(result, WorkerResult):
                result.latency_ms = max(result.latency_ms, latency_ms)
                return result
        if isinstance(outcome, WorkerResult):
            return outcome
        return WorkerResult(
            agent=agent,
            success=False,
            latency_ms=0,
            confidence=0.0,
            error="Unknown worker result type",
        )
