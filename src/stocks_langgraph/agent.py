from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .graph import build_graph
from .providers.fx import StaticFXProvider
from .providers.market_data import build_market_data_provider
from .schemas import AgentInput, ApprovalRequest, InteractionAnalytics
from .services.intent_parser import IntentParser
from .services.portfolio_service import PortfolioService
from .state import approval_from_control, build_interaction_analytics, hydrate_state
from .tracing import traceable


class AgentResult(BaseModel):
    thread_id: str
    response: str
    interrupted: bool = False
    approval_request: ApprovalRequest | None = None
    analytics: InteractionAnalytics | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class TradingAgent(AbstractContextManager["TradingAgent"]):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.prepare_paths()
        self.settings.configure_runtime_env()
        self.market_data = build_market_data_provider(self.settings)
        self.fx_provider = StaticFXProvider()
        self.intent_parser = IntentParser(self.settings, supported_symbols=self.market_data.supported_symbols)
        self.portfolio_service = PortfolioService()
        self._checkpointer_cm = SqliteSaver.from_conn_string(str(self.settings.checkpoint_db_path))
        self.checkpointer = self._checkpointer_cm.__enter__()
        self.graph = build_graph(
            market_data=self.market_data,
            fx_provider=self.fx_provider,
            intent_parser=self.intent_parser,
            portfolio_service=self.portfolio_service,
            checkpointer=self.checkpointer,
        )

    @classmethod
    def from_env(cls) -> "TradingAgent":
        return cls(get_settings())

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._checkpointer_cm.__exit__(None, None, None)

    @traceable(name="trading_agent_run_turn")
    def run_turn(
        self,
        thread_id: str,
        content: str,
        user_id: str | None = None,
        base_currency: str | None = None,
    ) -> AgentResult:
        pending_state = self.get_state(thread_id)
        pending_approval = approval_from_control(pending_state.get("control")) if pending_state else None
        if pending_approval and pending_approval.status == "PENDING":
            return AgentResult(
                thread_id=thread_id,
                response=(
                    f"Approval still pending for {pending_approval.action.lower()} {pending_approval.quantity:g} "
                    f"shares of {pending_approval.symbol}. Approve or reject it before sending a new message."
                ),
                interrupted=True,
                approval_request=pending_approval,
                analytics=self._build_analytics(pending_state),
                state=pending_state,
            )

        payload = AgentInput(
            thread_id=thread_id,
            user_id=user_id or self.settings.default_user_id,
            content=content,
            base_currency=(base_currency or self.settings.default_base_currency).upper(),
        )
        config = self._config(thread_id)
        raw_result = self.graph.invoke({"input": payload.model_dump(mode="json")}, config=config)
        state = self.get_state(thread_id)
        return self._build_result(thread_id, raw_result, state)

    @traceable(name="trading_agent_approve")
    def approve(
        self,
        thread_id: str,
        approved: bool,
        reviewer: str = "human-reviewer",
        reason: str | None = None,
    ) -> AgentResult:
        config = self._config(thread_id)
        raw_result = self.graph.invoke(
            Command(resume={"approved": approved, "reviewer": reviewer, "reason": reason}),
            config=config,
        )
        state = self.get_state(thread_id)
        return self._build_result(thread_id, raw_result, state)

    @traceable(name="trading_agent_get_state")
    def get_state(self, thread_id: str) -> dict[str, Any]:
        config = self._config(thread_id)
        try:
            snapshot = self.graph.get_state(config)
        except Exception:
            return {}
        if snapshot is None:
            return {}
        return snapshot.values or {}

    def _build_result(self, thread_id: str, raw_result: dict[str, Any], state: dict[str, Any]) -> AgentResult:
        interrupted = bool(raw_result.get("__interrupt__")) if isinstance(raw_result, dict) else False
        approval_request = approval_from_control(state.get("control")) if state else None
        response = state.get("response", "")
        if interrupted and approval_request and not response:
            response = (
                f"Approval required to {approval_request.action.lower()} {approval_request.quantity:g} "
                f"shares of {approval_request.symbol}."
            )
        return AgentResult(
            thread_id=thread_id,
            response=response or "No response available.",
            interrupted=interrupted,
            approval_request=approval_request,
            analytics=self._build_analytics(state),
            state=state,
        )

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _build_analytics(self, state: dict[str, Any]) -> InteractionAnalytics | None:
        if not state:
            return None
        bundle = hydrate_state(state)
        return build_interaction_analytics(
            bundle,
            market_data_provider=self.market_data.provider_name,
            intent_parser_mode=self._resolve_intent_parser_mode(state),
            langsmith_tracing_enabled=self.settings.langsmith_enabled,
            langsmith_project=self.settings.langsmith_project,
        )

    def _resolve_intent_parser_mode(self, state: dict[str, Any]) -> str:
        usage_entries = state.get("telemetry", {}).get("usage", [])
        if usage_entries:
            latest_usage = usage_entries[-1]
            model = latest_usage.get("model", "unknown")
            metadata = latest_usage.get("metadata", {})
            if model == "rule-based" and metadata.get("requested_parser") == "openrouter":
                return "openrouter-fallback"
            if model == "rule-based":
                return "rule-based"
            return f"openrouter:{model}"
        control_flags = state.get("control", {}).get("flags", [])
        if self.settings.openrouter_enabled and "intent_parse_error" in control_flags:
            return "openrouter-failed"
        return "openrouter" if self.settings.openrouter_enabled else "rule-based"
