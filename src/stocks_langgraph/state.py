from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from .schemas import (
    AgentInput,
    ApprovalDecision,
    ApprovalRequest,
    ConversationMessage,
    ConversationState,
    ControlState,
    DecisionEvent,
    DomainState,
    ExecutionState,
    InteractionAnalytics,
    TelemetryState,
    ToolEvent,
    TurnObservation,
    UsageRecord,
    UserProfile,
    new_id,
    utc_now,
)


class AgentState(TypedDict, total=False):
    input: dict[str, Any]
    execution: dict[str, Any]
    domain: dict[str, Any]
    conversation: dict[str, Any]
    control: dict[str, Any]
    telemetry: dict[str, Any]
    response: str


@dataclass
class StateBundle:
    agent_input: AgentInput | None
    execution: ExecutionState
    domain: DomainState
    conversation: ConversationState
    control: ControlState
    telemetry: TelemetryState
    response: str = ""


def hydrate_state(state: AgentState) -> StateBundle:
    agent_input = AgentInput.model_validate(state["input"]) if state.get("input") else None
    execution = ExecutionState.model_validate(state.get("execution", {}))
    domain = DomainState.model_validate(state.get("domain", {}))
    conversation = ConversationState.model_validate(state.get("conversation", {}))
    control = ControlState.model_validate(state.get("control", {}))
    telemetry = TelemetryState.model_validate(state.get("telemetry", {}))
    response = state.get("response", "")

    if agent_input:
        if not execution.session_id:
            execution.session_id = agent_input.thread_id
        if not execution.thread_id:
            execution.thread_id = agent_input.thread_id
        if not domain.user.user_id:
            domain.user = UserProfile(user_id=agent_input.user_id, base_currency=agent_input.base_currency)
        else:
            domain.user.base_currency = agent_input.base_currency

    return StateBundle(
        agent_input=agent_input,
        execution=execution,
        domain=domain,
        conversation=conversation,
        control=control,
        telemetry=telemetry,
        response=response,
    )


def dump_bundle(bundle: StateBundle) -> AgentState:
    payload: AgentState = {
        "execution": bundle.execution.model_dump(mode="json"),
        "domain": bundle.domain.model_dump(mode="json"),
        "conversation": bundle.conversation.model_dump(mode="json"),
        "control": bundle.control.model_dump(mode="json"),
        "telemetry": bundle.telemetry.model_dump(mode="json"),
        "response": bundle.response,
    }
    if bundle.agent_input:
        payload["input"] = bundle.agent_input.model_dump(mode="json")
    return payload


def enter_node(bundle: StateBundle, node_name: str) -> None:
    bundle.execution.current_node = node_name
    bundle.execution.step_count += 1
    bundle.execution.last_transition_at = utc_now()
    bundle.execution.route_history.append(node_name)
    bundle.execution.current_turn_routes.append(node_name)


def start_turn(bundle: StateBundle) -> None:
    bundle.execution.current_turn_id = new_id("turn")
    bundle.execution.current_turn_routes = []
    bundle.execution.next_node_hint = None
    bundle.execution.turn_started_at = utc_now()
    bundle.execution.turn_start_input_tokens = bundle.telemetry.total_input_tokens
    bundle.execution.turn_start_output_tokens = bundle.telemetry.total_output_tokens
    bundle.execution.turn_start_total_tokens = bundle.telemetry.total_tokens
    bundle.execution.turn_start_estimated_cost_usd = bundle.telemetry.estimated_cost_usd
    bundle.execution.turn_start_tool_event_count = len(bundle.telemetry.tool_events)
    bundle.execution.turn_start_decision_event_count = len(bundle.telemetry.decision_events)


def append_message(bundle: StateBundle, role: str, content: str) -> None:
    bundle.conversation.messages.append(ConversationMessage(role=role, content=content))


def compact_conversation(bundle: StateBundle, keep_last: int = 6) -> None:
    if len(bundle.conversation.messages) <= keep_last:
        return

    archived_messages = bundle.conversation.messages[:-keep_last]
    archived_summary = " | ".join(f"{message.role}: {message.content}" for message in archived_messages[-4:])
    if bundle.conversation.context_summary:
        bundle.conversation.context_summary = f"{bundle.conversation.context_summary} | {archived_summary}"
    else:
        bundle.conversation.context_summary = archived_summary
    bundle.conversation.messages = bundle.conversation.messages[-keep_last:]


def record_tool_event(bundle: StateBundle, event: ToolEvent) -> None:
    bundle.telemetry.tool_events.append(event)


def record_decision(
    bundle: StateBundle,
    node_name: str,
    decision_type: str,
    selected_route: str,
    rationale: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    bundle.telemetry.decision_events.append(
        DecisionEvent(
            node_name=node_name,
            decision_type=decision_type,
            selected_route=selected_route,
            rationale=rationale,
            metadata=metadata or {},
        )
    )
    bundle.execution.next_node_hint = selected_route


def record_usage(bundle: StateBundle, usage: UsageRecord | None) -> None:
    if usage is None:
        return
    bundle.telemetry.usage.append(usage)
    bundle.telemetry.total_input_tokens += usage.input_tokens
    bundle.telemetry.total_output_tokens += usage.output_tokens
    bundle.telemetry.total_tokens += usage.total_tokens
    bundle.telemetry.estimated_cost_usd += usage.estimated_cost_usd


def set_error(bundle: StateBundle, message: str, flag: str | None = None) -> None:
    bundle.execution.last_error = message
    bundle.response = message
    if flag and flag not in bundle.control.flags:
        bundle.control.flags.append(flag)


def build_interaction_analytics(
    bundle: StateBundle,
    market_data_provider: str,
    intent_parser_mode: str,
    langsmith_tracing_enabled: bool,
    langsmith_project: str,
) -> InteractionAnalytics:
    approval_pending = bool(
        bundle.control.pending_approval and bundle.control.pending_approval.status == "PENDING"
    )
    return InteractionAnalytics(
        route_path=list(bundle.execution.current_turn_routes),
        decision_count=len(bundle.telemetry.decision_events) - bundle.execution.turn_start_decision_event_count,
        tool_call_count=len(bundle.telemetry.tool_events) - bundle.execution.turn_start_tool_event_count,
        input_tokens=bundle.telemetry.total_input_tokens - bundle.execution.turn_start_input_tokens,
        output_tokens=bundle.telemetry.total_output_tokens - bundle.execution.turn_start_output_tokens,
        total_tokens=bundle.telemetry.total_tokens - bundle.execution.turn_start_total_tokens,
        estimated_cost_usd=round(
            bundle.telemetry.estimated_cost_usd - bundle.execution.turn_start_estimated_cost_usd,
            8,
        ),
        approval_pending=approval_pending,
        market_data_provider=market_data_provider,
        intent_parser_mode=intent_parser_mode,
        langsmith_tracing_enabled=langsmith_tracing_enabled,
        langsmith_project=langsmith_project,
    )


def record_turn_observation(bundle: StateBundle) -> TurnObservation | None:
    user_message = bundle.agent_input.content if bundle.agent_input else ""
    if not user_message:
        return None

    pending_approval = bundle.control.pending_approval
    route_path = list(bundle.execution.current_turn_routes)
    approval_status = pending_approval.status if pending_approval else None
    if bundle.control.last_decision and any(node in route_path for node in ("approval_request", "human_approval")):
        approval_status = "APPROVED" if bundle.control.last_decision.approved else "REJECTED"
    elif not any(node in route_path for node in ("approval_request", "human_approval")):
        approval_status = None

    observation = TurnObservation(
        turn_id=bundle.execution.current_turn_id or new_id("turn"),
        thread_id=bundle.execution.thread_id or bundle.execution.session_id,
        user_message=user_message,
        assistant_response=bundle.response,
        route_path=route_path,
        decision_count=len(bundle.telemetry.decision_events) - bundle.execution.turn_start_decision_event_count,
        tool_call_count=len(bundle.telemetry.tool_events) - bundle.execution.turn_start_tool_event_count,
        interrupted=bool(pending_approval and pending_approval.status == "PENDING"),
        approval_status=approval_status,
        input_tokens=bundle.telemetry.total_input_tokens - bundle.execution.turn_start_input_tokens,
        output_tokens=bundle.telemetry.total_output_tokens - bundle.execution.turn_start_output_tokens,
        total_tokens=bundle.telemetry.total_tokens - bundle.execution.turn_start_total_tokens,
        estimated_cost_usd=round(
            bundle.telemetry.estimated_cost_usd - bundle.execution.turn_start_estimated_cost_usd,
            8,
        ),
    )
    bundle.telemetry.turns.append(observation)
    return observation


def read_next_route(bundle: StateBundle, default_route: str = "finalize") -> str:
    return bundle.execution.next_node_hint or default_route


def approval_from_control(control_payload: dict[str, Any] | None) -> ApprovalRequest | None:
    if not control_payload:
        return None
    pending = control_payload.get("pending_approval")
    return ApprovalRequest.model_validate(pending) if pending else None


def decision_from_control(control_payload: dict[str, Any] | None) -> ApprovalDecision | None:
    if not control_payload:
        return None
    decision = control_payload.get("last_decision")
    return ApprovalDecision.model_validate(decision) if decision else None
