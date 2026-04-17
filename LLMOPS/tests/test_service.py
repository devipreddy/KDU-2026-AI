from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixit_llmops.llm_provider import MockLLMProvider
from fixit_llmops.models import SupportRequest
from fixit_llmops.service import FixItService


class ServiceTests(unittest.TestCase):
    def test_end_to_end_request_uses_mock_provider(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox = Path(tmp_dir)
            (sandbox / "config").mkdir()
            (sandbox / "prompts").mkdir()
            (sandbox / "prompts" / "faq").mkdir()
            (sandbox / "prompts" / "booking").mkdir()
            (sandbox / "prompts" / "complaint").mkdir()
            (sandbox / "prompts" / "fallback").mkdir()
            (sandbox / "config" / "app.yaml").write_text((project_root / "config" / "app.yaml").read_text(encoding="utf-8"), encoding="utf-8")
            for relative in [
                "prompts/faq/v1.yaml",
                "prompts/faq/v2.yaml",
                "prompts/booking/v1.yaml",
                "prompts/complaint/v1.yaml",
                "prompts/complaint/v2.yaml",
                "prompts/fallback/v1.yaml",
            ]:
                target = sandbox / relative
                target.write_text((project_root / relative).read_text(encoding="utf-8"), encoding="utf-8")

            service = FixItService(sandbox / "config" / "app.yaml", provider_override=MockLLMProvider())
            response = service.process(SupportRequest(query="Can I reschedule my cleaning appointment?"))

        self.assertEqual(response.category, "booking")
        self.assertEqual(response.selected_model_alias, "low")
        self.assertGreater(response.estimated_cost_usd, 0)


if __name__ == "__main__":
    unittest.main()
