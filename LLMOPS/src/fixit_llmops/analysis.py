from __future__ import annotations

from .cost import CostCalculator
from .models import FixItConfig, TokenUsage


class CostAnalyzer:
    def __init__(self, config: FixItConfig) -> None:
        self.config = config
        self.calculator = CostCalculator()

    def projected_monthly_cost(self) -> float:
        monthly_queries = self.config.analysis.monthly_queries
        total = 0.0
        for alias, share in self.config.analysis.projected_query_mix.items():
            tokens = self.config.analysis.average_tokens[alias]
            usage = TokenUsage(prompt_tokens=tokens.prompt, completion_tokens=tokens.completion)
            cost_per_request = self.calculator.estimate(self.config.models[alias], usage)
            total += monthly_queries * share * cost_per_request
        return round(total, 2)

    def legacy_monthly_cost(self) -> float:
        return round(self.config.analysis.legacy_all_high_cost_usd, 2)

    def savings_amount(self) -> float:
        return round(self.legacy_monthly_cost() - self.projected_monthly_cost(), 2)

    def savings_percent(self) -> float:
        legacy = self.legacy_monthly_cost()
        if legacy <= 0:
            return 0.0
        return round((self.savings_amount() / legacy) * 100, 2)

