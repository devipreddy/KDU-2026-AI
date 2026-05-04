from __future__ import annotations

from app.core.models import SessionSnapshot, TriageDecision
from app.services.openai_provider import OpenAIProvider


class TriageAgent:
    name = "triage"

    def __init__(self, provider: OpenAIProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def route(self, transcript: str, snapshot: SessionSnapshot) -> tuple[TriageDecision, dict | None]:
        system_prompt = (
            "You are the triage agent for a voice customer service system. "
            "Decide whether the user should stay with triage or be handed to billing. "
            "Prefer billing for invoices, refunds, duplicate charges, payment methods, cards, plan pricing, "
            "renewals, balances, and receipts. Return JSON only."
        )
        recent = "\n".join(f"{m.role}: {m.text}" for m in snapshot.recent_messages[-4:])
        user_prompt = (
            f"Conversation summary: {snapshot.conversation_summary or 'none'}\n"
            f"Recent messages:\n{recent or 'none'}\n"
            f"Latest transcript: {transcript}\n"
            "Classify intent, extract entities, choose target_agent, and explain the routing_reason."
        )
        data, usage = await self.provider.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="triage_decision",
            schema=TriageDecision.model_json_schema(),
            model=self.model,
        )
        return TriageDecision.model_validate(data), usage.model_dump() if usage else None
