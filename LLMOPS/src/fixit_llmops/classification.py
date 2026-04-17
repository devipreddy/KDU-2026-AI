from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ClassificationConfig, ClassificationResult


@dataclass(slots=True)
class CategoryScore:
    name: str
    score: float
    matched_rules: list[str]


class QueryClassifier:
    def __init__(self, config: ClassificationConfig) -> None:
        self.config = config

    def classify(self, query: str) -> ClassificationResult:
        normalized = query.lower().strip()
        scores = [self._score_category(name, rule, normalized) for name, rule in self.config.categories.items()]
        scores.sort(key=lambda item: (item.score, self._category_priority(item.name)), reverse=True)
        best = scores[0] if scores else CategoryScore(self.config.fallback_category, 0.0, [])
        runner_up = scores[1] if len(scores) > 1 else CategoryScore(self.config.fallback_category, 0.0, [])

        if best.score <= 0:
            category = self.config.fallback_category
            matched_rules: list[str] = []
            confidence = 0.25
        else:
            category = best.name
            matched_rules = best.matched_rules
            confidence = min(0.99, 0.55 + min(best.score, 4.0) * 0.08 + max(best.score - runner_up.score, 0) * 0.05)

        complexity = self._infer_complexity(category, normalized)
        category_rule = self.config.categories.get(category)
        response_type = category_rule.response_type if category_rule else "standard"
        return ClassificationResult(
            category=category,
            complexity=complexity,
            response_type=response_type,
            confidence=round(confidence, 2),
            matched_rules=matched_rules,
        )

    def _score_category(self, name: str, rule, normalized: str) -> CategoryScore:
        score = 0.0
        matched_rules: list[str] = []
        for keyword in rule.keywords:
            if keyword.lower() in normalized:
                score += 1.0
                matched_rules.append(f"keyword:{keyword}")
        for pattern in rule.regexes:
            if re.search(pattern, normalized):
                score += 1.5
                matched_rules.append(f"regex:{pattern}")
        return CategoryScore(name=name, score=score, matched_rules=matched_rules)

    def _infer_complexity(self, category: str, normalized: str) -> str:
        word_count = len(normalized.split())
        category_rule = self.config.categories.get(category)
        default_complexity = category_rule.default_complexity if category_rule else "medium"

        if any(token in normalized for token in self.config.escalation_keywords):
            return "high"
        if category == "complaint":
            return "high"
        if word_count >= self.config.very_long_query_word_threshold:
            return "high"
        if word_count >= self.config.long_query_word_threshold:
            return "medium"
        if any(token in normalized for token in self.config.medium_keywords):
            return "medium"
        return default_complexity

    @staticmethod
    def _category_priority(name: str) -> int:
        priorities = {"complaint": 3, "booking": 2, "faq": 1, "fallback": 0}
        return priorities.get(name, 0)

