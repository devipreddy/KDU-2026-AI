from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .models import BudgetConfig, BudgetDecision, ModelSpec, TokenUsage
from .storage import SQLiteStateStore


class CostCalculator:
    @staticmethod
    def estimate(model: ModelSpec, usage: TokenUsage) -> float:
        prompt_cost = (usage.prompt_tokens / 1_000_000) * model.input_cost_per_1m
        completion_cost = (usage.completion_tokens / 1_000_000) * model.output_cost_per_1m
        return round(prompt_cost + completion_cost, 6)


class BudgetGuard:
    def __init__(self, budget_config: BudgetConfig, store: SQLiteStateStore, timezone_name: str) -> None:
        self.budget_config = budget_config
        self.store = store
        self.timezone = ZoneInfo(timezone_name)

    def assess(self, requested_model_alias: str, low_cost_alias: str, now: datetime | None = None) -> BudgetDecision:
        moment = now.astimezone(self.timezone) if now else datetime.now(self.timezone)
        day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        month_start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if moment.month == 12:
            month_end = moment.replace(year=moment.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            month_end = moment.replace(month=moment.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

        daily_spend = self.store.get_spend_between(
            day_start.astimezone(timezone.utc).isoformat(),
            day_end.astimezone(timezone.utc).isoformat(),
        )
        monthly_spend = self.store.get_spend_between(
            month_start.astimezone(timezone.utc).isoformat(),
            month_end.astimezone(timezone.utc).isoformat(),
        )
        daily_ratio = daily_spend / self.budget_config.daily_limit_usd if self.budget_config.daily_limit_usd else 0.0
        monthly_ratio = monthly_spend / self.budget_config.monthly_limit_usd if self.budget_config.monthly_limit_usd else 0.0
        ratio = max(daily_ratio, monthly_ratio)

        if ratio >= self.budget_config.hard_stop_ratio:
            if requested_model_alias != low_cost_alias:
                return self._decision(
                    selected_model_alias=low_cost_alias,
                    action="forced_low_budget",
                    reason="Budget hard stop reached for premium traffic.",
                    daily_spend=daily_spend,
                    monthly_spend=monthly_spend,
                )
            return self._decision(
                selected_model_alias=None,
                action="safe_fallback_only",
                reason="Budget hard stop reached for all model traffic.",
                daily_spend=daily_spend,
                monthly_spend=monthly_spend,
            )

        if requested_model_alias != low_cost_alias and ratio >= self.budget_config.degrade_ratio:
            return self._decision(
                selected_model_alias=low_cost_alias,
                action="degraded_to_low_budget",
                reason="Budget guardrail triggered downgrade to low-cost model.",
                daily_spend=daily_spend,
                monthly_spend=monthly_spend,
            )

        if ratio >= self.budget_config.warn_ratio:
            return self._decision(
                selected_model_alias=requested_model_alias,
                action="warning_budget",
                reason="Budget warning threshold reached.",
                daily_spend=daily_spend,
                monthly_spend=monthly_spend,
            )

        return self._decision(
            selected_model_alias=requested_model_alias,
            action="proceed",
            reason=None,
            daily_spend=daily_spend,
            monthly_spend=monthly_spend,
        )

    def _decision(
        self,
        selected_model_alias: str | None,
        action: str,
        reason: str | None,
        daily_spend: float,
        monthly_spend: float,
    ) -> BudgetDecision:
        return BudgetDecision(
            selected_model_alias=selected_model_alias,
            action=action,
            reason=reason,
            daily_spend_usd=round(daily_spend, 6),
            monthly_spend_usd=round(monthly_spend, 6),
            daily_remaining_usd=round(max(self.budget_config.daily_limit_usd - daily_spend, 0.0), 6),
            monthly_remaining_usd=round(max(self.budget_config.monthly_limit_usd - monthly_spend, 0.0), 6),
        )
