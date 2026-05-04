from __future__ import annotations

from app.core.models import BillingReply, ConsensusResult, HandoffState
from app.services.openai_provider import OpenAIProvider


class BillingAgent:
    name = "billing"

    def __init__(self, provider: OpenAIProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def respond(
        self,
        handoff: HandoffState,
        consensus: ConsensusResult,
    ) -> tuple[BillingReply, dict | None]:
        system_prompt = (
            "You are the billing agent in a live voice support system. "
            "Answer in 2-4 short sentences that sound natural when spoken aloud. "
            "Be direct, cite only information present in the context, and avoid promising actions you cannot take. "
            "If the data is partial, say that clearly and offer a safe next step. Return JSON only."
        )
        recent = "\n".join(f"{m.role}: {m.text}" for m in handoff.recent_messages[-4:])
        user_prompt = (
            f"User query: {handoff.user_query}\n"
            f"Intent: {handoff.current_intent}\n"
            f"Entities: {handoff.entities}\n"
            f"Conversation summary: {handoff.conversation_summary or 'none'}\n"
            f"Recent messages:\n{recent or 'none'}\n"
            f"Consensus context: {consensus.answer_context}\n"
            f"Conflict notes: {consensus.conflict_notes}\n"
            f"Confidence: {consensus.confidence}\n"
            "Generate a concise spoken reply, a short summary, follow_up_required, and actions."
        )
        data, usage = await self.provider.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="billing_reply",
            schema=BillingReply.model_json_schema(),
            model=self.model,
        )
        return BillingReply.model_validate(data), usage.model_dump() if usage else None
