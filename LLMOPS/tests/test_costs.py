from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fixit_llmops.cost import BudgetGuard, CostCalculator
from fixit_llmops.models import BudgetConfig, ModelSpec, RequestLogRecord, TokenUsage
from fixit_llmops.storage import SQLiteStateStore


class CostTests(unittest.TestCase):
    def test_cost_calculation_uses_token_pricing(self) -> None:
        model = ModelSpec(model_id="test", input_cost_per_1m=0.4, output_cost_per_1m=1.6)
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = CostCalculator().estimate(model, usage)
        self.assertAlmostEqual(cost, 0.0012)

    def test_budget_guard_downgrades_high_model_when_threshold_is_crossed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SQLiteStateStore(Path(tmp_dir) / "state.sqlite3")
            now = datetime(2026, 4, 17, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
            store.record_request(
                RequestLogRecord(
                    request_id="one",
                    timestamp=now,
                    query="x",
                    category="faq",
                    complexity="low",
                    response_type="simple",
                    requested_model_alias="high",
                    used_model_alias="high",
                    model_id="test/high",
                    prompt_id="p",
                    prompt_version="v1",
                    prompt_tokens=1,
                    completion_tokens=1,
                    estimated_cost_usd=91.0,
                    budget_action="proceed",
                    classification_confidence=0.9,
                    response_preview="ok",
                )
            )
            guard = BudgetGuard(BudgetConfig(monthly_limit_usd=100, daily_limit_usd=100, degrade_ratio=0.9), store, "Asia/Kolkata")
            decision = guard.assess("high", "low", now=now)

        self.assertEqual(decision.selected_model_alias, "low")
        self.assertEqual(decision.action, "degraded_to_low_budget")

    def test_budget_guard_blocks_all_when_low_budget_is_also_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SQLiteStateStore(Path(tmp_dir) / "state.sqlite3")
            now = datetime(2026, 4, 17, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
            store.record_request(
                RequestLogRecord(
                    request_id="one",
                    timestamp=now,
                    query="x",
                    category="faq",
                    complexity="low",
                    response_type="simple",
                    requested_model_alias="low",
                    used_model_alias="low",
                    model_id="test/low",
                    prompt_id="p",
                    prompt_version="v1",
                    prompt_tokens=1,
                    completion_tokens=1,
                    estimated_cost_usd=101.0,
                    budget_action="proceed",
                    classification_confidence=0.9,
                    response_preview="ok",
                )
            )
            guard = BudgetGuard(BudgetConfig(monthly_limit_usd=100, daily_limit_usd=100), store, "Asia/Kolkata")
            decision = guard.assess("low", "low", now=now)

        self.assertIsNone(decision.selected_model_alias)
        self.assertEqual(decision.action, "safe_fallback_only")


if __name__ == "__main__":
    unittest.main()

