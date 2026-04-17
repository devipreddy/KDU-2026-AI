from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixit_llmops.models import PromptConfig
from fixit_llmops.prompts import PromptManager


class PromptManagerTests(unittest.TestCase):
    def test_latest_stable_prefers_highest_stable_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_root = Path(tmp_dir) / "prompts" / "faq"
            prompt_root.mkdir(parents=True, exist_ok=True)
            (prompt_root / "v1.yaml").write_text(
                """
prompt_id: fixit.faq
category: faq
version: v1
status: stable
selection_score: 0.5
template: "hello {category}"
""",
                encoding="utf-8",
            )
            (prompt_root / "v2.yaml").write_text(
                """
prompt_id: fixit.faq
category: faq
version: v2
status: stable
selection_score: 0.4
template: "hi {category}"
""",
                encoding="utf-8",
            )
            manager = PromptManager(PromptConfig(root_dir="prompts", selection_strategy="latest_stable"), Path(tmp_dir))
            prompt = manager.get_prompt("faq")
            rendered = manager.render(prompt, {"category": "faq"})

        self.assertEqual(prompt.version, "v2")
        self.assertEqual(rendered, "hi faq")

    def test_best_performing_uses_selection_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_root = Path(tmp_dir) / "prompts" / "faq"
            prompt_root.mkdir(parents=True, exist_ok=True)
            (prompt_root / "v1.yaml").write_text(
                """
prompt_id: fixit.faq
category: faq
version: v1
status: stable
selection_score: 0.6
template: "v1"
""",
                encoding="utf-8",
            )
            (prompt_root / "v2.yaml").write_text(
                """
prompt_id: fixit.faq
category: faq
version: v2
status: stable
selection_score: 0.9
template: "v2"
""",
                encoding="utf-8",
            )
            manager = PromptManager(PromptConfig(root_dir="prompts", selection_strategy="best_performing"), Path(tmp_dir))
            prompt = manager.get_prompt("faq")

        self.assertEqual(prompt.version, "v2")


if __name__ == "__main__":
    unittest.main()

