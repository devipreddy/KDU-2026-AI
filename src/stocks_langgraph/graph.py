from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .providers.fx import FXProvider
from .providers.market_data import MarketDataError, MarketDataProvider
from .schemas import ApprovalDecision, ApprovalRequest, MarketQuote, ToolEvent
from .services.intent_parser import IntentParser
from .services.portfolio_service import PortfolioService
from .state import (
    AgentState,
    append_message,
    compact_conversation,
    dump_bundle,
    enter_node,
    hydrate_state,
    read_next_route,
    record_decision,
    record_turn_observation,
    record_tool_event,
    record_usage,
    set_error,
    start_turn,
)

BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "reveal system prompt",
    "skip approval",
    "bypass approval",
)


def build_graph(
    market_data: MarketDataProvider,
    fx_provider: FXProvider,
    intent_parser: IntentParser,
    portfolio_service: PortfolioService,
    checkpointer: Any,
):
    def bootstrap_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        if bundle.agent_input:
            start_turn(bundle)
        enter_node(bundle, "bootstrap")
        bundle.execution.last_error = None
        bundle.response = ""
        bundle.control.blocked = False
        bundle.control.flags = []
        if bundle.control.pending_approval and bundle.control.pending_approval.status != "PENDING":
            bundle.control.pending_approval = None
        if bundle.agent_input:
            append_message(bundle, "user", bundle.agent_input.content)
            compact_conversation(bundle)
        return dump_bundle(bundle)

    def guard_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "guard")
        latest_text = bundle.conversation.messages[-1].content if bundle.conversation.messages else ""
        lowered = latest_text.lower()
        if any(pattern in lowered for pattern in BLOCKED_PATTERNS):
            bundle.control.blocked = True
            bundle.control.flags.append("prompt_injection_blocked")
            bundle.response = "Unsafe instruction detected and blocked. Please submit a clean trading request."
            record_decision(
                bundle,
                node_name="guard",
                decision_type="safety_gate",
                selected_route="finalize",
                rationale="Prompt injection pattern detected in user content.",
            )
        else:
            record_decision(
                bundle,
                node_name="guard",
                decision_type="safety_gate",
                selected_route="intent",
                rationale="Input passed guard checks.",
            )
        return dump_bundle(bundle)

    def intent_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "intent")
        if bundle.control.blocked:
            return dump_bundle(bundle)

        latest_text = bundle.conversation.messages[-1].content if bundle.conversation.messages else ""
        try:
            intent, usage = intent_parser.parse(latest_text, bundle.domain.user.base_currency)
        except Exception as exc:
            set_error(bundle, str(exc), "intent_parse_error")
            record_decision(
                bundle,
                node_name="intent",
                decision_type="intent_route",
                selected_route="finalize",
                rationale="Intent parsing failed before routing.",
                metadata={"error": str(exc)},
            )
            return dump_bundle(bundle)
        bundle.conversation.intent = intent
        record_usage(bundle, usage)

        if intent.action in {"BUY", "SELL"} and not intent.symbol:
            bundle.response = "Please include a stock symbol for the trade request."
        elif intent.action in {"BUY", "SELL"} and not intent.quantity:
            bundle.response = "Please include a quantity for the trade request."
        elif intent.action == "QUOTE" and not intent.symbol:
            bundle.response = "Please include a stock symbol for the quote request."
        elif intent.action == "UNKNOWN":
            bundle.response = (
                "I could not classify the request. Try formats like 'Quote SAP.DE', "
                "'Buy 2 RELIANCE.NS', or 'Show my portfolio'."
            )

        next_route = "finalize"
        rationale = "Request could not continue and was finalized."
        if not bundle.response and intent is not None:
            if intent.action == "PORTFOLIO":
                next_route = "portfolio"
                rationale = "Intent mapped to a portfolio summary request."
            elif intent.action in {"BUY", "SELL", "QUOTE"}:
                next_route = "price"
                rationale = f"Intent '{intent.action}' requires a market quote."
        record_decision(
            bundle,
            node_name="intent",
            decision_type="intent_route",
            selected_route=next_route,
            rationale=rationale,
            metadata={"intent": intent.model_dump(mode="json") if intent is not None else {}},
        )
        return dump_bundle(bundle)

    def portfolio_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "portfolio")
        bundle.domain = portfolio_service.recalculate(bundle.domain)
        bundle.response = portfolio_service.build_summary(bundle.domain)
        return dump_bundle(bundle)

    def price_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "price")
        intent = bundle.conversation.intent
        if intent is None or intent.symbol is None:
            set_error(bundle, "Cannot fetch a quote because the symbol is missing.", "missing_symbol")
            return dump_bundle(bundle)

        try:
            quote = market_data.get_quote(intent.symbol)
            if quote.currency_native == bundle.domain.user.base_currency:
                quote.base_currency = bundle.domain.user.base_currency
                quote.price_base = quote.price_native
            bundle.domain.latest_quote = quote
            record_tool_event(
                bundle,
                ToolEvent(
                    tool_name="market_data",
                    succeeded=True,
                    input_payload={"symbol": intent.symbol},
                    output_payload=quote.model_dump(mode="json"),
                ),
            )
        except MarketDataError as exc:
            set_error(bundle, str(exc), "market_data_error")
            record_tool_event(
                bundle,
                ToolEvent(
                    tool_name="market_data",
                    succeeded=False,
                    input_payload={"symbol": intent.symbol},
                    error=str(exc),
                ),
            )

        next_route = "finalize"
        rationale = "Quote lookup failed or no quote was produced."
        quote = bundle.domain.latest_quote
        if not bundle.execution.last_error and quote is not None and intent is not None:
            if quote.currency_native != bundle.domain.user.base_currency:
                next_route = "fx"
                rationale = "Quote currency differs from the portfolio base currency."
            elif intent.action == "QUOTE":
                next_route = "finalize"
                rationale = "Quote request can return immediately without approval."
            else:
                next_route = "approval_request"
                rationale = "Trade request has a normalized quote and must enter approval."
        record_decision(
            bundle,
            node_name="price",
            decision_type="quote_route",
            selected_route=next_route,
            rationale=rationale,
            metadata={"symbol": intent.symbol if intent else None},
        )
        return dump_bundle(bundle)

    def fx_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "fx")
        quote = bundle.domain.latest_quote
        if quote is None:
            set_error(bundle, "No quote available for FX conversion.", "missing_quote")
            return dump_bundle(bundle)

        try:
            fx_rate = fx_provider.get_rate(quote.currency_native, bundle.domain.user.base_currency)
            quote.fx_rate = fx_rate
            quote.base_currency = bundle.domain.user.base_currency
            quote.price_base = round(quote.price_native * fx_rate, 4)
            bundle.domain.latest_quote = quote
            record_tool_event(
                bundle,
                ToolEvent(
                    tool_name="fx_conversion",
                    succeeded=True,
                    input_payload={
                        "amount": quote.price_native,
                        "from_currency": quote.currency_native,
                        "to_currency": bundle.domain.user.base_currency,
                    },
                    output_payload={"fx_rate": fx_rate, "price_base": quote.price_base},
                ),
            )
        except Exception as exc:
            set_error(bundle, f"FX conversion failed: {exc}", "fx_conversion_error")
            record_tool_event(
                bundle,
                ToolEvent(
                    tool_name="fx_conversion",
                    succeeded=False,
                    input_payload={
                        "from_currency": quote.currency_native,
                        "to_currency": bundle.domain.user.base_currency,
                    },
                    error=str(exc),
                ),
            )

        intent = bundle.conversation.intent
        if bundle.execution.last_error:
            next_route = "finalize"
            rationale = "FX conversion failed, so the workflow finalized with an error."
        elif intent is not None and intent.action == "QUOTE":
            next_route = "finalize"
            rationale = "Converted quote is ready to return to the user."
        else:
            next_route = "approval_request"
            rationale = "Trade quote is now normalized and can enter approval."
        record_decision(
            bundle,
            node_name="fx",
            decision_type="fx_route",
            selected_route=next_route,
            rationale=rationale,
        )
        return dump_bundle(bundle)

    def approval_request_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "approval_request")
        intent = bundle.conversation.intent
        quote = bundle.domain.latest_quote

        if intent is None or quote is None:
            set_error(bundle, "Cannot raise approval without a validated intent and quote.", "approval_setup_error")
            record_decision(
                bundle,
                node_name="approval_request",
                decision_type="approval_route",
                selected_route="finalize",
                rationale="Approval prerequisites were incomplete.",
            )
            return dump_bundle(bundle)

        validation_error = portfolio_service.validate_order(bundle.domain, intent, quote)
        if validation_error:
            set_error(bundle, validation_error, "validation_error")
            record_decision(
                bundle,
                node_name="approval_request",
                decision_type="approval_route",
                selected_route="finalize",
                rationale="Trade validation failed before approval.",
                metadata={"validation_error": validation_error},
            )
            return dump_bundle(bundle)

        notional = round((intent.quantity or 0) * (quote.price_base or 0), 4)
        request = ApprovalRequest(
            action=intent.action,
            symbol=intent.symbol or "",
            quantity=intent.quantity or 0,
            estimated_notional_base=notional,
            base_currency=bundle.domain.user.base_currency,
        )
        bundle.control.pending_approval = request
        bundle.control.requires_human_input = True
        bundle.response = (
            f"Approval required to {request.action.lower()} {request.quantity:g} shares of {request.symbol} "
            f"for approximately {request.estimated_notional_base:.2f} {request.base_currency}."
        )
        record_decision(
            bundle,
            node_name="approval_request",
            decision_type="approval_route",
            selected_route="human_approval",
            rationale="Sensitive trade action requires explicit human approval.",
            metadata={"approval_id": request.approval_id},
        )
        return dump_bundle(bundle)

    def human_approval_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "human_approval")
        request = bundle.control.pending_approval
        if request is None:
            set_error(bundle, "No pending approval request was found.", "approval_missing")
            return dump_bundle(bundle)

        decision_payload = interrupt({"approval_request": request.model_dump(mode="json")})
        decision = ApprovalDecision(
            approval_id=request.approval_id,
            approved=bool(decision_payload.get("approved")),
            reviewer=decision_payload.get("reviewer", "human-reviewer"),
            reason=decision_payload.get("reason"),
        )
        request.status = "APPROVED" if decision.approved else "REJECTED"
        bundle.control.pending_approval = request
        bundle.control.last_decision = decision
        bundle.control.requires_human_input = False
        if not decision.approved:
            bundle.response = f"Order rejected by {decision.reviewer}. No trade was executed."
        record_decision(
            bundle,
            node_name="human_approval",
            decision_type="human_decision",
            selected_route="execute_order" if decision.approved else "finalize",
            rationale="Human reviewer approved the order." if decision.approved else "Human reviewer rejected the order.",
            metadata={"reviewer": decision.reviewer, "approved": decision.approved},
        )
        return dump_bundle(bundle)

    def execute_order_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "execute_order")
        intent = bundle.conversation.intent
        quote = bundle.domain.latest_quote
        request = bundle.control.pending_approval

        if intent is None or quote is None or request is None:
            set_error(bundle, "Execution prerequisites are incomplete.", "execution_error")
            return dump_bundle(bundle)

        bundle.domain = portfolio_service.apply_order(bundle.domain, intent, quote)
        bundle.response = (
            f"{intent.action} order executed for {intent.quantity:g} shares of {intent.symbol} "
            f"at {quote.price_base:.2f} {bundle.domain.user.base_currency} each. "
            f"Notional: {(intent.quantity or 0) * (quote.price_base or 0):.2f} {bundle.domain.user.base_currency}."
        )
        return dump_bundle(bundle)

    def finalize_node(state: AgentState) -> AgentState:
        bundle = hydrate_state(state)
        enter_node(bundle, "finalize")
        if not bundle.response:
            bundle.response = build_response(bundle.domain.latest_quote, bundle)
        append_message(bundle, "assistant", bundle.response)
        compact_conversation(bundle)
        record_turn_observation(bundle)
        bundle.agent_input = None
        return dump_bundle(bundle)

    def route_after_guard(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "intent")

    def route_after_intent(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "finalize")

    def route_after_price(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "finalize")

    def route_after_fx(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "finalize")

    def route_after_approval_request(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "finalize")

    def route_after_human_approval(state: AgentState) -> str:
        bundle = hydrate_state(state)
        return read_next_route(bundle, "finalize")

    builder = StateGraph(AgentState)
    builder.add_node("bootstrap", bootstrap_node)
    builder.add_node("guard", guard_node)
    builder.add_node("intent", intent_node)
    builder.add_node("portfolio", portfolio_node)
    builder.add_node("price", price_node)
    builder.add_node("fx", fx_node)
    builder.add_node("approval_request", approval_request_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_order", execute_order_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "guard")
    builder.add_conditional_edges("guard", route_after_guard)
    builder.add_conditional_edges("intent", route_after_intent)
    builder.add_edge("portfolio", "finalize")
    builder.add_conditional_edges("price", route_after_price)
    builder.add_conditional_edges("fx", route_after_fx)
    builder.add_conditional_edges("approval_request", route_after_approval_request)
    builder.add_conditional_edges("human_approval", route_after_human_approval)
    builder.add_edge("execute_order", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def build_response(quote: MarketQuote | None, bundle: Any) -> str:
    intent = bundle.conversation.intent
    if bundle.response:
        return bundle.response
    if bundle.control.blocked:
        return "Unsafe request blocked."
    if intent is None:
        return "No actionable intent was found."
    if intent.action == "PORTFOLIO":
        return "Portfolio summary is unavailable."
    if intent.action == "QUOTE" and quote is not None:
        if quote.price_base is not None and quote.base_currency is not None and quote.currency_native != quote.base_currency:
            return (
                f"Latest quote for {quote.symbol}: {quote.price_native:.2f} {quote.currency_native} "
                f"(~{quote.price_base:.2f} {quote.base_currency} at FX {quote.fx_rate:.4f})."
            )
        return f"Latest quote for {quote.symbol}: {quote.price_native:.2f} {quote.currency_native}."
    if intent.action in {"BUY", "SELL"} and bundle.control.last_decision and not bundle.control.last_decision.approved:
        return "Trade rejected during human review."
    return "Workflow completed."
