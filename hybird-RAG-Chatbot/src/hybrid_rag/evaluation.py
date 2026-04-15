from __future__ import annotations

from typing import Any

from .constants import UNKNOWN_ANSWER
from .utils import keyword_tokenize


class ResponseEvaluator:
    def __init__(self, confidence_threshold: float, min_context_coverage: float) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_context_coverage = min_context_coverage

    def evaluate(
        self,
        query: str,
        answer: str,
        selected_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not selected_context:
            return {
                "confidence": 0.0,
                "needs_disclaimer": True,
                "details": {
                    "selected_docs": 0,
                    "answer_relevance": 0.0,
                    "answer_faithfulness": 0.0,
                    "context_coverage": 0.0,
                    "avg_combined_score": 0.0,
                },
            }

        combined_context = " ".join(doc["page_content"] for doc in selected_context)
        answer_relevance = self._overlap_ratio(query, answer)
        answer_faithfulness = self._overlap_ratio(answer, combined_context)
        context_coverage = max(doc.get("context_coverage", 0.0) for doc in selected_context)
        avg_combined_score = sum(doc.get("combined_score", 0.0) for doc in selected_context) / len(selected_context)

        confidence = (
            (0.25 * answer_relevance)
            + (0.35 * answer_faithfulness)
            + (0.25 * context_coverage)
            + (0.15 * min(max(avg_combined_score / 5, 0.0), 1.0))
        )
        confidence = max(0.0, min(1.0, confidence))

        if answer.strip() == UNKNOWN_ANSWER:
            confidence = min(confidence, 0.35)

        needs_disclaimer = (
            confidence < self.confidence_threshold
            or context_coverage < self.min_context_coverage
            or answer_faithfulness < 0.45
        )

        return {
            "confidence": confidence,
            "needs_disclaimer": needs_disclaimer,
            "details": {
                "selected_docs": len(selected_context),
                "answer_relevance": answer_relevance,
                "answer_faithfulness": answer_faithfulness,
                "context_coverage": context_coverage,
                "avg_combined_score": avg_combined_score,
            },
        }

    def _overlap_ratio(self, source_text: str, target_text: str) -> float:
        source_tokens = set(keyword_tokenize(source_text))
        if not source_tokens:
            return 0.0
        target_tokens = set(keyword_tokenize(target_text))
        if not target_tokens:
            return 0.0
        return len(source_tokens.intersection(target_tokens)) / len(source_tokens)
