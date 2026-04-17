from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fixit_llmops.config import ConfigManager


class ConfigManagerTests(unittest.TestCase):
    def test_loads_yaml_and_expands_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "app.yaml"
            config_path.write_text(
                """
app:
  name: test-app
llm:
  base_url: ${TEST_BASE_URL:https://example.com}
  api_key_env: TEST_API_KEY
models:
  low:
    model_id: test/low
    input_cost_per_1m: 0.1
    output_cost_per_1m: 0.2
  high:
    model_id: test/high
    input_cost_per_1m: 1.0
    output_cost_per_1m: 2.0
budget:
  monthly_limit_usd: 100
feature_flags: {}
prompts:
  root_dir: prompts
classification:
  categories:
    faq:
      response_type: simple
      default_complexity: low
  escalation_keywords: []
  medium_keywords: []
routing:
  complexity_to_model:
    low: low
    medium: low
    high: high
""",
                encoding="utf-8",
            )
            os.environ["TEST_BASE_URL"] = "https://openrouter.example/api/v1"
            config = ConfigManager(config_path).load(force=True)

        self.assertEqual(config.llm.base_url, "https://openrouter.example/api/v1")
        self.assertEqual(config.models["low"].model_id, "test/low")
        self.assertEqual(config.budget.daily_limit_usd, round(100 / 30.0, 2))


if __name__ == "__main__":
    unittest.main()

