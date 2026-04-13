from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import TradingAgent
from .schemas import new_id
from .tracing import traceable


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_turns(*states: dict[str, Any]) -> list[dict[str, Any]]:
    seen_turn_ids: set[str] = set()
    turns: list[dict[str, Any]] = []
    for state in states:
        telemetry = state.get("telemetry", {})
        for turn in telemetry.get("turns", []):
            turn_id = turn.get("turn_id")
            if turn_id and turn_id not in seen_turn_ids:
                seen_turn_ids.add(turn_id)
                turns.append(turn)
    return turns


def _collect_decision_count(*states: dict[str, Any]) -> int:
    return sum(len(state.get("telemetry", {}).get("decision_events", [])) for state in states)


def _collect_tool_count(*states: dict[str, Any]) -> int:
    return sum(len(state.get("telemetry", {}).get("tool_events", [])) for state in states)


@traceable(name="evaluation_suite")
def run_evaluation_suite(agent: TradingAgent, output_path: Path | None = None) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    run_prefix = new_id("eval")
    quote_thread = f"{run_prefix}-quote"
    buy_thread = f"{run_prefix}-buy"
    empty_thread = f"{run_prefix}-empty"

    quote_result = agent.run_turn(
        thread_id=quote_thread,
        content="Quote SAP.DE",
        user_id="eval-user",
        base_currency="INR",
    )
    quote_state = quote_result.state
    quote_payload = quote_state.get("domain", {}).get("latest_quote", {})
    scenarios.append(
        {
            "name": "quote_with_fx_routing",
            "success": (
                quote_result.response.lower().startswith("latest quote")
                and quote_payload.get("currency_native") == "EUR"
                and quote_payload.get("base_currency") == "INR"
                and quote_payload.get("price_base") is not None
            ),
            "response": quote_result.response,
            "tool_events": quote_state.get("telemetry", {}).get("tool_events", []),
            "analytics": quote_result.analytics.model_dump(mode="json") if quote_result.analytics else {},
        }
    )

    buy_result = agent.run_turn(
        thread_id=buy_thread,
        content="Buy 2 SAP.DE",
        user_id="eval-user",
        base_currency="INR",
    )
    approved_result = agent.approve(buy_thread, approved=True, reviewer="evaluation-suite", reason="demo approval")
    approved_state = approved_result.state
    holdings = approved_state.get("domain", {}).get("portfolio", {}).get("holdings", {})
    scenarios.append(
        {
            "name": "buy_requires_human_approval",
            "success": (
                buy_result.interrupted
                and approved_result.response.lower().startswith("buy order executed")
                and holdings.get("SAP.DE", {}).get("quantity") == 2.0
            ),
            "approval_required": buy_result.interrupted,
            "approval_response": buy_result.response,
            "execution_response": approved_result.response,
            "analytics": approved_result.analytics.model_dump(mode="json") if approved_result.analytics else {},
        }
    )

    portfolio_result = agent.run_turn(
        thread_id=buy_thread,
        content="Show my portfolio",
        user_id="eval-user",
        base_currency="INR",
    )
    other_thread_result = agent.run_turn(
        thread_id=empty_thread,
        content="Show my portfolio",
        user_id="eval-user",
        base_currency="INR",
    )
    scenarios.append(
        {
            "name": "checkpoint_thread_isolation",
            "success": (
                "SAP.DE" in portfolio_result.response
                and "Holdings: none" in other_thread_result.response
            ),
            "stateful_thread_response": portfolio_result.response,
            "isolated_thread_response": other_thread_result.response,
            "analytics": portfolio_result.analytics.model_dump(mode="json") if portfolio_result.analytics else {},
        }
    )

    buy_thread_state = agent.get_state(buy_thread)
    quote_thread_state = agent.get_state(quote_thread)
    empty_thread_state = agent.get_state(empty_thread)
    turns = _collect_turns(quote_thread_state, buy_thread_state, empty_thread_state)
    total_input_tokens = sum(int(turn.get("input_tokens", 0)) for turn in turns)
    total_output_tokens = sum(int(turn.get("output_tokens", 0)) for turn in turns)
    total_tokens = sum(int(turn.get("total_tokens", 0)) for turn in turns)
    total_cost = round(sum(float(turn.get("estimated_cost_usd", 0.0)) for turn in turns), 8)
    decision_count = _collect_decision_count(quote_thread_state, buy_thread_state, empty_thread_state)
    tool_count = _collect_tool_count(quote_thread_state, buy_thread_state, empty_thread_state)

    observability = {
        "langsmith": {
            "configured": agent.settings.langsmith_enabled,
            "tracing_flag_enabled": agent.settings.langsmith_tracing,
            "project": agent.settings.langsmith_project,
            "status": "active" if agent.settings.langsmith_enabled else "awaiting_credentials",
            "note": (
                "Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to emit hosted traces."
                if not agent.settings.langsmith_enabled
                else "Hosted LangSmith traces should be emitted for graph and traceable spans."
            ),
        },
        "runtime": {
            "intent_parser_mode": quote_result.analytics.intent_parser_mode if quote_result.analytics else (
                "openrouter" if agent.settings.openrouter_enabled else "rule-based"
            ),
            "market_data_provider": agent.market_data.provider_name,
            "intent_parser_fallback_allowed": agent.settings.intent_parser_allow_fallback,
        },
    }

    summary = {
        "total_scenarios": len(scenarios),
        "passed_scenarios": sum(1 for scenario in scenarios if scenario["success"]),
        "pass_rate": round(sum(1 for scenario in scenarios if scenario["success"]) / len(scenarios), 4),
        "total_turn_observations": len(turns),
        "total_decision_events": decision_count,
        "total_tool_events": tool_count,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_cost_usd": total_cost,
        "langsmith_tracing_enabled": agent.settings.langsmith_enabled,
    }
    report = {
        "generated_at": _utc_iso(),
        "observability": observability,
        "scenarios": scenarios,
        "turn_observations": turns,
        "summary": summary,
    }

    destination = output_path or agent.settings.artifacts_dir / "evaluation-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    analytics_report = {
        "generated_at": report["generated_at"],
        "observability": observability,
        "token_usage_observations": turns,
        "summary": summary,
    }
    analytics_destination = destination.parent / "analytics-report.json"
    analytics_destination.write_text(json.dumps(analytics_report, indent=2, default=str), encoding="utf-8")
    return report
