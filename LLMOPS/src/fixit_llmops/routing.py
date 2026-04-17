from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from .classification import QueryClassifier
from .cost import BudgetGuard, CostCalculator
from .llm_provider import LLMProvider
from .models import FixItConfig, RequestLogRecord, SupportRequest, SupportResponse, TokenUsage
from .prompts import PromptManager
from .storage import SQLiteStateStore


class QueryRouter:
    def __init__(
        self,
        config: FixItConfig,
        classifier: QueryClassifier,
        prompt_manager: PromptManager,
        provider: LLMProvider,
        store: SQLiteStateStore,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.prompt_manager = prompt_manager
        self.provider = provider
        self.store = store
        self.cost_calculator = CostCalculator()
        self.budget_guard = BudgetGuard(config.budget, store, config.app.timezone)

    def handle(self, request: SupportRequest) -> SupportResponse:
        started = time.perf_counter()
        request_id = request.request_id or str(uuid4())
        classification = self.classifier.classify(request.query)
        desired_model_alias = self.config.routing.complexity_to_model.get(
            classification.complexity,
            self.config.routing.default_model_alias,
        )

        prompt_asset = self.prompt_manager.get_prompt(classification.category)
        prompt_text = self.prompt_manager.render(
            prompt_asset,
            {
                "category": classification.category,
                "complexity": classification.complexity,
                "response_type": classification.response_type,
            },
        )
        user_prompt = self._build_user_prompt(request.query, classification.category, classification.complexity)

        budget_action = "proceed"
        fallback_reason = None
        selected_model_alias = desired_model_alias
        if self.config.feature_flags.enable_budget_guardrails:
            budget_decision = self.budget_guard.assess(
                requested_model_alias=desired_model_alias,
                low_cost_alias=self.config.routing.default_model_alias,
            )
            budget_action = budget_decision.action
            fallback_reason = budget_decision.reason
            selected_model_alias = budget_decision.selected_model_alias
        else:
            budget_decision = self.budget_guard.assess(
                requested_model_alias=desired_model_alias,
                low_cost_alias=self.config.routing.default_model_alias,
            )
            budget_action = "proceed"
            fallback_reason = None

        if selected_model_alias is None:
            reply = self._safe_fallback_reply(request.query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._record_request(
                request_id=request_id,
                request=request,
                classification=classification,
                desired_model_alias=desired_model_alias,
                used_model_alias=None,
                model_id=None,
                prompt_asset=prompt_asset,
                usage=TokenUsage(),
                estimated_cost_usd=0.0,
                budget_action=budget_action,
                reply=reply,
            )
            return SupportResponse(
                request_id=request_id,
                reply=reply,
                category=classification.category,
                complexity=classification.complexity,
                response_type=classification.response_type,
                selected_model_alias=None,
                selected_model_id=None,
                prompt_id=prompt_asset.prompt_id,
                prompt_version=prompt_asset.version,
                estimated_cost_usd=0.0,
                budget_action=budget_action,
                fallback_reason=fallback_reason,
                latency_ms=latency_ms,
            )

        generation = None
        model_fallback_reason = fallback_reason
        aliases_to_try = [selected_model_alias]
        if self.config.feature_flags.enable_model_fallbacks:
            aliases_to_try.extend(self.config.routing.fallback_order.get(selected_model_alias, []))

        for alias in aliases_to_try:
            model_spec = self.config.models[alias]
            try:
                generation = self.provider.generate(alias, model_spec, prompt_text, user_prompt)
                selected_model_alias = alias
                break
            except Exception as exc:  # pragma: no cover - exercised by live integrations
                model_fallback_reason = f"{model_fallback_reason or ''} Provider error on {alias}: {exc}".strip()
                generation = None

        if generation is None:
            reply = self._safe_fallback_reply(request.query)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._record_request(
                request_id=request_id,
                request=request,
                classification=classification,
                desired_model_alias=desired_model_alias,
                used_model_alias=None,
                model_id=None,
                prompt_asset=prompt_asset,
                usage=TokenUsage(),
                estimated_cost_usd=0.0,
                budget_action="provider_fallback",
                reply=reply,
            )
            return SupportResponse(
                request_id=request_id,
                reply=reply,
                category=classification.category,
                complexity=classification.complexity,
                response_type=classification.response_type,
                selected_model_alias=None,
                selected_model_id=None,
                prompt_id=prompt_asset.prompt_id,
                prompt_version=prompt_asset.version,
                estimated_cost_usd=0.0,
                budget_action="provider_fallback",
                fallback_reason=model_fallback_reason or "All configured providers failed.",
                latency_ms=latency_ms,
            )

        estimated_cost = self.cost_calculator.estimate(self.config.models[selected_model_alias], generation.usage)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.store.record_prompt_use(prompt_asset.prompt_id, prompt_asset.version, estimated_cost, success=True)
        self._record_request(
            request_id=request_id,
            request=request,
            classification=classification,
            desired_model_alias=desired_model_alias,
            used_model_alias=selected_model_alias,
            model_id=generation.model_id,
            prompt_asset=prompt_asset,
            usage=generation.usage,
            estimated_cost_usd=estimated_cost,
            budget_action=budget_action,
            reply=generation.text,
        )
        return SupportResponse(
            request_id=request_id,
            reply=generation.text,
            category=classification.category,
            complexity=classification.complexity,
            response_type=classification.response_type,
            selected_model_alias=selected_model_alias,
            selected_model_id=generation.model_id,
            prompt_id=prompt_asset.prompt_id,
            prompt_version=prompt_asset.version,
            estimated_cost_usd=estimated_cost,
            budget_action=budget_action,
            fallback_reason=model_fallback_reason,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _build_user_prompt(query: str, category: str, complexity: str) -> str:
        return (
            "Customer support query for FixIt.\n"
            f"Predicted category: {category}\n"
            f"Predicted complexity: {complexity}\n"
            f"Customer message: {query}"
        )

    @staticmethod
    def _safe_fallback_reply(query: str) -> str:
        return (
            "I can help with that, but I need to hand this off carefully. "
            f"A support agent can continue from your message: \"{query[:120]}\"."
        )

    def _record_request(
        self,
        request_id: str,
        request: SupportRequest,
        classification,
        desired_model_alias: str,
        used_model_alias: str | None,
        model_id: str | None,
        prompt_asset,
        usage: TokenUsage,
        estimated_cost_usd: float,
        budget_action: str,
        reply: str,
    ) -> None:
        if not self.config.feature_flags.enable_request_logging:
            return

        record = RequestLogRecord(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            query=request.query,
            category=classification.category,
            complexity=classification.complexity,
            response_type=classification.response_type,
            requested_model_alias=desired_model_alias,
            used_model_alias=used_model_alias,
            model_id=model_id,
            prompt_id=prompt_asset.prompt_id,
            prompt_version=prompt_asset.version,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            budget_action=budget_action,
            classification_confidence=classification.confidence,
            response_preview=reply[:250],
        )
        self.store.record_request(record)
