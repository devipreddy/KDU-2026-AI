from __future__ import annotations

from app.config import Settings
from app.schemas import CostBreakdown, TokenUsage


def build_cost_breakdown(tokens: TokenUsage, settings: Settings) -> CostBreakdown:
    amount = (
        (tokens.input / 1_000_000) * settings.model_input_cost_per_million
        + (tokens.output / 1_000_000) * settings.model_output_cost_per_million
    )
    return CostBreakdown(amount=round(amount, 6), currency=settings.model_currency)
