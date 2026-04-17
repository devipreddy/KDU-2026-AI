from __future__ import annotations

import unittest
from pathlib import Path

from fixit_llmops.classification import QueryClassifier
from fixit_llmops.config import ConfigManager


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cls.config = ConfigManager(project_root / "config" / "app.yaml").load(force=True)
        cls.classifier = QueryClassifier(cls.config.classification)

    def test_faq_query_maps_to_low_complexity(self) -> None:
        result = self.classifier.classify("What are your hours?")
        self.assertEqual(result.category, "faq")
        self.assertEqual(result.complexity, "low")
        self.assertEqual(result.response_type, "simple")

    def test_booking_query_maps_to_medium_complexity(self) -> None:
        result = self.classifier.classify("Can I reschedule my cleaning appointment?")
        self.assertEqual(result.category, "booking")
        self.assertEqual(result.complexity, "medium")
        self.assertEqual(result.response_type, "standard")

    def test_complaint_query_maps_to_high_complexity(self) -> None:
        result = self.classifier.classify("My plumber didn't show up, need refund")
        self.assertEqual(result.category, "complaint")
        self.assertEqual(result.complexity, "high")
        self.assertEqual(result.response_type, "complex")


if __name__ == "__main__":
    unittest.main()

