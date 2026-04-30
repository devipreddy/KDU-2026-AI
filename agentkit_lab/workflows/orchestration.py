"""Phase 2 and 3 multi-agent orchestration with strict tool isolation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from ..memory.fact_extractor import FactExtractor
from ..memory.manager import StructuredMemoryManager
from ..runtime.cost_control import (
    JsonFileCache,
    ModelRouter,
    TokenBudgetManager,
    stable_cache_key,
)
from ..runtime.observability import ObservabilityManager
from ..schemas import (
    CapabilityContract,
    CompressedToolSignature,
    ContextTTLItem,
    CoordinationReport,
    FinanceHandoffData,
    FinanceHandoffPayload,
    FinanceRequest,
    HRHandoffData,
    HRHandoffPayload,
    HRRequest,
    PromptSegments,
    RoutingDecision,
    to_plain_data,
)
from ..sdk_support import (
    RunContextWrapper,
    bind_run_context_wrapper,
    configure_provider,
    import_agents_sdk,
)
from ..services.finance import FinanceService
from ..services.hr import HRService
from ..settings import AppSettings

COORDINATOR_SYSTEM_SEGMENT = (
    "You are an orchestration-only manager. "
    "Never answer finance or HR questions directly. "
    "Only delegate to specialists whose capability contracts explicitly allow the requested work. "
    "If a capability appears in an agent's cannot_do list, never route to that agent. "
    "When delegating, use only the structured payload in CONTEXT.handoff_payloads for that domain. "
    "Do not pass raw transcript text, prior chat history, or unrelated fields. "
    "Aggregate specialist outputs into one concise final answer."
)

SPECIALIST_SYSTEM_SEGMENTS = {
    "finance": (
        "You are the Finance Agent. "
        "Only handle finance capabilities allowed by your contract. "
        "Never answer HR questions."
    ),
    "hr": (
        "You are the HR Agent. "
        "Only handle HR capabilities allowed by your contract. "
        "Never answer finance questions."
    ),
}

CAPABILITY_KEYWORDS = {
    "salary": ["salary", "compensation", "pay"],
    "transactions": ["transaction", "transactions", "expense", "reimbursement"],
    "banking_details": ["bank", "banking", "routing", "account", "deposit"],
    "pto": ["pto", "paid time off", "time off"],
    "leave_balance": ["leave balance", "vacation", "vacation balance", "leave"],
}

PROMPT_SEGMENT_CACHE: dict[tuple[str, ...], tuple[str, str]] = {}


@dataclass(slots=True)
class CoordinatorContext:
    finance_service: FinanceService
    hr_service: HRService
    memory_manager: StructuredMemoryManager
    session_id: str
    user_input: str = ""
    delegation_sequence: list[str] = field(default_factory=list)
    payload_log: list[dict[str, Any]] = field(default_factory=list)
    route_decisions: list[RoutingDecision] = field(default_factory=list)
    distilled_payloads: dict[str, Any] = field(default_factory=dict)
    context_ttl_store: dict[str, dict[str, ContextTTLItem]] = field(default_factory=dict)
    tool_cache: JsonFileCache | None = None
    observability: ObservabilityManager | None = None
    token_budget_manager: TokenBudgetManager | None = None
    last_context_budget_report: dict[str, Any] = field(default_factory=dict)
    model_routing: dict[str, str] = field(default_factory=dict)


def _build_capability_contracts() -> dict[str, CapabilityContract]:
    return {
        "finance": CapabilityContract(
            agent="finance",
            can_do=["salary", "transactions", "banking_details"],
            cannot_do=["pto", "leave_balance"],
            delegation_tool="finance_specialist",
        ),
        "hr": CapabilityContract(
            agent="hr",
            can_do=["pto", "leave_balance"],
            cannot_do=["salary", "transactions", "banking_details"],
            delegation_tool="hr_specialist",
        ),
    }


def _build_tool_signatures() -> dict[str, CompressedToolSignature]:
    return {
        "finance": CompressedToolSignature(
            name="finance_specialist",
            args=[
                "intent",
                "data.employee_name?",
                "data.routing_number?",
                "data.account_number?",
                "data.account_holder_name?",
            ],
            summary="salary, transactions, banking_details",
        ),
        "hr": CompressedToolSignature(
            name="hr_specialist",
            args=["intent", "data.employee_name?"],
            summary="pto, leave_balance",
        ),
    }


def _extract_requested_capabilities(user_input: str) -> dict[str, list[int]]:
    lowered = user_input.lower()
    matches: dict[str, list[int]] = {}
    for capability, keywords in CAPABILITY_KEYWORDS.items():
        positions = [lowered.find(keyword) for keyword in keywords if keyword in lowered]
        positions = [position for position in positions if position >= 0]
        if positions:
            matches[capability] = sorted(set(positions))
    return matches


def _extract_employee_name(user_input: str) -> str | None:
    match = re.search(r"\b([A-Z][a-z]+)'s\b", user_input)
    if match:
        return match.group(1)
    ignored_tokens = {
        "What",
        "How",
        "Update",
        "Please",
        "Count",
        "Tell",
        "Routing",
        "Banking",
        "Account",
        "Salary",
        "PTO",
    }
    matches = re.findall(r"\b([A-Z][a-z]+)\b", user_input)
    for token in matches:
        if token not in ignored_tokens:
            return token
    return None


def _route_domains_from_contracts(
    user_input: str,
    contracts: dict[str, CapabilityContract],
) -> list[RoutingDecision]:
    matched_capabilities = _extract_requested_capabilities(user_input)
    decisions: list[RoutingDecision] = []
    for domain, contract in contracts.items():
        domain_matches = [
            capability
            for capability in matched_capabilities
            if capability in contract.can_do
        ]
        if not domain_matches:
            continue
        blocked = [
            capability
            for capability in matched_capabilities
            if capability in contract.cannot_do
        ]
        if blocked and len(domain_matches) == 0:
            continue
        positions = [
            position
            for capability in domain_matches
            for position in matched_capabilities[capability]
        ]
        decisions.append(
            RoutingDecision(
                domain=domain,
                matched_capabilities=sorted(domain_matches),
                matched_positions=sorted(positions),
            )
        )

    if not decisions:
        return [
            RoutingDecision(domain="finance"),
            RoutingDecision(domain="hr"),
        ]

    decisions.sort(
        key=lambda decision: (
            decision.matched_positions[0] if decision.matched_positions else 10**6,
            decision.domain,
        )
    )
    return decisions


def _build_structured_context(
    route_decisions: list[RoutingDecision],
    handoff_payloads: dict[str, Any],
) -> dict[str, Any]:
    return {
        "employee_name": next(
            (
                payload.data.employee_name
                for payload in handoff_payloads.values()
                if getattr(payload, "data", None) is not None
                and getattr(payload.data, "employee_name", None)
            ),
            None,
        ),
        "requested_capabilities": sorted(
            {
                capability
                for decision in route_decisions
                for capability in decision.matched_capabilities
            }
        ),
        "route_domains": [decision.domain for decision in route_decisions],
        "handoff_payloads": {
            domain: to_plain_data(payload) for domain, payload in handoff_payloads.items()
        },
    }


def _get_cached_prompt_segments(
    route_decisions: list[RoutingDecision],
    contracts: dict[str, CapabilityContract],
    tool_signatures: dict[str, CompressedToolSignature],
    user_input: str,
) -> PromptSegments:
    active_domains = tuple(decision.domain for decision in route_decisions)
    if active_domains not in PROMPT_SEGMENT_CACHE:
        tools_lines = []
        for domain in active_domains:
            contract = contracts[domain]
            signature = tool_signatures[domain]
            tools_lines.append(
                f"contract::{contract.agent} can_do={contract.can_do} cannot_do={contract.cannot_do}"
            )
            tools_lines.append(
                f"tool::{signature.name} args={signature.args} summary={signature.summary}"
            )
        PROMPT_SEGMENT_CACHE[active_domains] = (
            COORDINATOR_SYSTEM_SEGMENT,
            "\n".join(tools_lines),
        )

    system_segment, tools_segment = PROMPT_SEGMENT_CACHE[active_domains]
    return PromptSegments(
        system=system_segment,
        tools=tools_segment,
        context="",
        user=user_input,
    )


def _extract_intent_for_domain(decision: RoutingDecision) -> str:
    capabilities = set(decision.matched_capabilities)
    if decision.domain == "finance":
        if "banking_details" in capabilities:
            return "update_bank_details"
        if "salary" in capabilities:
            return "get_salary"
        if "transactions" in capabilities:
            return "get_transactions"
        return "finance_request"
    if decision.domain == "hr":
        if "pto" in capabilities or "leave_balance" in capabilities:
            return "get_pto_balance"
        return "hr_request"
    return "unknown_request"


def _extract_entities(user_input: str) -> dict[str, Any]:
    extraction = FactExtractor().extract(user_input)
    banking_update = extraction.banking_update
    return {
        "employee_name": _extract_employee_name(user_input),
        "routing_number": (
            banking_update.routing_number if banking_update is not None else None
        ),
        "account_number": (
            banking_update.account_number if banking_update is not None else None
        ),
        "account_holder_name": None,
    }


def _apply_relevance_filter(
    domain: str,
    intent: str,
    entities: dict[str, Any],
) -> dict[str, Any]:
    if domain == "finance" and intent == "update_bank_details":
        allowed = [
            "employee_name",
            "routing_number",
            "account_number",
            "account_holder_name",
        ]
        return {
            key: entities[key]
            for key in allowed
            if entities.get(key) is not None
        }
    if domain == "finance" and intent == "get_salary":
        return {
            "employee_name": entities["employee_name"]
        } if entities.get("employee_name") else {}
    if domain == "hr" and intent == "get_pto_balance":
        return {
            "employee_name": entities["employee_name"]
        } if entities.get("employee_name") else {}
    return {}


def _build_handoff_payloads(
    user_input: str,
    route_decisions: list[RoutingDecision],
) -> dict[str, FinanceHandoffPayload | HRHandoffPayload]:
    entities = _extract_entities(user_input)
    payloads: dict[str, FinanceHandoffPayload | HRHandoffPayload] = {}

    for decision in route_decisions:
        intent = _extract_intent_for_domain(decision)
        filtered_entities = _apply_relevance_filter(
            decision.domain,
            intent,
            entities,
        )
        if decision.domain == "finance":
            payloads["finance"] = FinanceHandoffPayload(
                intent=intent,
                data=FinanceHandoffData(**filtered_entities),
            )
        elif decision.domain == "hr":
            payloads["hr"] = HRHandoffPayload(
                intent=intent,
                data=HRHandoffData(**filtered_entities),
            )

    return payloads


def _seed_context_ttl_store(
    handoff_payloads: dict[str, FinanceHandoffPayload | HRHandoffPayload],
    *,
    default_ttl: int,
) -> dict[str, dict[str, ContextTTLItem]]:
    store: dict[str, dict[str, ContextTTLItem]] = {}
    for domain, payload in handoff_payloads.items():
        payload_data = to_plain_data(payload.data)
        store[domain] = {
            key: ContextTTLItem(value=value, ttl=default_ttl)
            for key, value in payload_data.items()
            if value is not None
        }
    return store


def _active_handoff_payloads(
    handoff_payloads: dict[str, FinanceHandoffPayload | HRHandoffPayload],
    ttl_store: dict[str, dict[str, ContextTTLItem]],
) -> dict[str, FinanceHandoffPayload | HRHandoffPayload]:
    active: dict[str, FinanceHandoffPayload | HRHandoffPayload] = {}
    for domain, payload in handoff_payloads.items():
        active_fields = {
            key: item.value
            for key, item in ttl_store.get(domain, {}).items()
            if item.ttl > 0
        }
        if domain == "finance":
            active[domain] = FinanceHandoffPayload(
                intent=payload.intent,
                data=FinanceHandoffData(**active_fields),
            )
        elif domain == "hr":
            active[domain] = HRHandoffPayload(
                intent=payload.intent,
                data=HRHandoffData(**active_fields),
            )
    return active


def _serialize_ttl_state(
    ttl_store: dict[str, dict[str, ContextTTLItem]],
) -> dict[str, dict[str, int]]:
    return {
        domain: {key: item.ttl for key, item in items.items()}
        for domain, items in ttl_store.items()
    }


def _consume_context_ttl(
    ttl_store: dict[str, dict[str, ContextTTLItem]],
    domain: str,
) -> None:
    for item in ttl_store.get(domain, {}).values():
        item.ttl = max(0, item.ttl - 1)


def _build_coordinator_input_filter(prompt_segments: PromptSegments):
    def _filter(data):
        try:
            from agents.run import ModelInputData
        except ImportError:
            @dataclass(slots=True)
            class ModelInputData:  # type: ignore[no-redef]
                input: Any
                instructions: str | None = None

        active_payloads = _active_handoff_payloads(
            data.context.distilled_payloads,
            data.context.context_ttl_store,
        )
        context_payload = _build_structured_context(
            data.context.route_decisions,
            active_payloads,
        )
        token_budget_manager = getattr(data.context, "token_budget_manager", None)
        observability = getattr(data.context, "observability", None)
        if token_budget_manager is not None:
            context_payload, budget_report = token_budget_manager.compress_context(
                context_payload
            )
            data.context.last_context_budget_report = budget_report.as_dict()
            if observability is not None:
                observability.record_tokens(
                    prompt_tokens=budget_report.final_estimated_tokens
                )
        return ModelInputData(
            input=(
                "CONTEXT: "
                + json.dumps(context_payload, sort_keys=True)
                + f"\nUSER: {data.context.user_input}"
            ),
            instructions=prompt_segments.system + "\n" + prompt_segments.tools,
        )

    return _filter


def _coordinator_execution_input(context: CoordinatorContext) -> str:
    active_payloads = _active_handoff_payloads(
        context.distilled_payloads,
        context.context_ttl_store,
    )
    payload = _build_structured_context(context.route_decisions, active_payloads)
    token_budget_manager = getattr(context, "token_budget_manager", None)
    observability = getattr(context, "observability", None)
    if token_budget_manager is not None:
        payload, budget_report = token_budget_manager.compress_context(payload)
        context.last_context_budget_report = budget_report.as_dict()
        if observability is not None:
            observability.record_tokens(
                prompt_tokens=budget_report.final_estimated_tokens
            )
    return "CONTEXT: " + json.dumps(payload, sort_keys=True) + f"\nUSER: {context.user_input}"


def _normalize_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return to_plain_data(value)
    if isinstance(value, dict):
        return value
    return {"value": value}


def _record_payload(ctx, domain: str) -> dict[str, Any]:
    payload = _normalize_payload(getattr(ctx, "tool_input", None))
    if payload:
        ctx.context.payload_log.append({"domain": domain, "payload": payload})
    return payload


def _build_finance_agent(settings: AppSettings):
    agents = import_agents_sdk()
    bind_run_context_wrapper(globals())

    @agents.function_tool
    def lookup_salary(
        ctx: RunContextWrapper[CoordinatorContext],
        employee_name: str,
    ) -> str:
        """Look up an employee salary in the finance system."""

        started_at = time.perf_counter()
        payload = _record_payload(ctx, "finance")
        payload_data = payload.get("data", payload)
        resolved_name = employee_name or payload_data.get("employee_name")
        cache = ctx.context.tool_cache
        observability = ctx.context.observability
        cache_key = stable_cache_key(
            "finance:get_salary",
            {"employee_name": resolved_name},
        )

        if cache is not None:
            lookup = cache.get(cache_key)
            if lookup.hit:
                if observability is not None:
                    observability.record_tool_cache_hit()
                    observability.trace(
                        agent="finance",
                        tool="get_salary",
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        status="cache_hit",
                        details={"employee_name": resolved_name},
                    )
                return str(lookup.value)

        try:
            result = ctx.context.finance_service.get_salary(resolved_name)
        except Exception as exc:
            if observability is not None:
                observability.trace(
                    agent="finance",
                    tool="get_salary",
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status="failure",
                    details={"employee_name": resolved_name, "error": str(exc)},
                )
            raise

        if cache is not None:
            cache.set(cache_key, result)
        if observability is not None:
            observability.trace(
                agent="finance",
                tool="get_salary",
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                details={"employee_name": resolved_name},
            )
        return result

    @agents.function_tool
    def apply_banking_update(
        ctx: RunContextWrapper[CoordinatorContext],
        employee_name: str | None = None,
        routing_number: str | None = None,
        account_number: str | None = None,
        account_holder_name: str | None = None,
    ) -> str:
        """Store and validate banking details for a finance update."""

        started_at = time.perf_counter()
        payload = _record_payload(ctx, "finance")
        payload_data = payload.get("data", payload)
        request = FinanceRequest(
            intent="update_banking_details",
            employee_name=employee_name or payload_data.get("employee_name"),
            routing_number=routing_number or payload_data.get("routing_number"),
            account_number=account_number or payload_data.get("account_number"),
            account_holder_name=account_holder_name or payload_data.get("account_holder_name"),
        )

        memory_fragments = []
        if request.routing_number:
            memory_fragments.append(f"routing number {request.routing_number}")
        if request.account_number:
            memory_fragments.append(f"account number {request.account_number}")
        if request.account_holder_name:
            memory_fragments.append(f"account holder {request.account_holder_name}")
        if memory_fragments:
            ctx.context.memory_manager.ingest_message(
                ctx.context.session_id,
                "user",
                ". ".join(memory_fragments),
            )

        service_result = ctx.context.finance_service.update_banking_details(request)
        facts = ctx.context.memory_manager.get_case_facts(ctx.context.session_id)
        if facts.requires_user_input:
            if ctx.context.observability is not None:
                ctx.context.observability.alert(
                    alert_type="missing_required_fields",
                    severity="warning",
                    message="Banking update requires additional user input.",
                    context={"missing_fields": list(facts.missing_fields)},
                )
                ctx.context.observability.trace(
                    agent="finance",
                    tool="update_bank_details",
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status="requires_user_input",
                    details={"missing_fields": list(facts.missing_fields)},
                )
            return (
                f"{service_result} The case is marked requires_user_input with missing fields: "
                f"{', '.join(facts.missing_fields)}."
            )
        if ctx.context.observability is not None:
            ctx.context.observability.trace(
                agent="finance",
                tool="update_bank_details",
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                details={"employee_name": request.employee_name},
            )
        return service_result

    return agents.Agent(
        name="Finance Agent",
        model=settings.finance_model,
        instructions=(
            SPECIALIST_SYSTEM_SEGMENTS["finance"]
            + " Tools: lookup_salary(employee_name), "
            + "apply_banking_update(employee_name, routing_number?, account_number?, account_holder_name?). "
            + "Use the distilled handoff payload only."
        ),
        tools=[lookup_salary, apply_banking_update],
    )


def _build_hr_agent(settings: AppSettings):
    agents = import_agents_sdk()
    bind_run_context_wrapper(globals())

    @agents.function_tool
    def lookup_pto_balance(
        ctx: RunContextWrapper[CoordinatorContext],
        employee_name: str,
    ) -> str:
        """Look up PTO balances in the HR system."""

        started_at = time.perf_counter()
        payload = _record_payload(ctx, "hr")
        payload_data = payload.get("data", payload)
        resolved_name = employee_name or payload_data.get("employee_name")
        cache = ctx.context.tool_cache
        observability = ctx.context.observability
        cache_key = stable_cache_key(
            "hr:get_pto_balance",
            {"employee_name": resolved_name},
        )

        if cache is not None:
            lookup = cache.get(cache_key)
            if lookup.hit:
                if observability is not None:
                    observability.record_tool_cache_hit()
                    observability.trace(
                        agent="hr",
                        tool="get_pto_balance",
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        status="cache_hit",
                        details={"employee_name": resolved_name},
                    )
                return str(lookup.value)

        try:
            result = ctx.context.hr_service.get_pto_balance(resolved_name)
        except Exception as exc:
            if observability is not None:
                observability.trace(
                    agent="hr",
                    tool="get_pto_balance",
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status="failure",
                    details={"employee_name": resolved_name, "error": str(exc)},
                )
            raise

        if cache is not None:
            cache.set(cache_key, result)
        if observability is not None:
            observability.trace(
                agent="hr",
                tool="get_pto_balance",
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                details={"employee_name": resolved_name},
            )
        return result

    return agents.Agent(
        name="HR Agent",
        model=settings.hr_model,
        instructions=(
            SPECIALIST_SYSTEM_SEGMENTS["hr"]
            + " Tools: lookup_pto_balance(employee_name)."
        ),
        tools=[lookup_pto_balance],
    )


def build_specialist_registry(settings: AppSettings) -> dict[str, Any]:
    finance_agent = _build_finance_agent(settings)
    hr_agent = _build_hr_agent(settings)

    def finance_input_builder(payload: FinanceHandoffPayload) -> str:
        return json.dumps(to_plain_data(payload), sort_keys=True)

    def hr_input_builder(payload: HRHandoffPayload) -> str:
        return json.dumps(to_plain_data(payload), sort_keys=True)

    return {
        "finance": finance_agent.as_tool(
            tool_name="finance_specialist",
            tool_description="intent + distilled finance data",
            parameters=FinanceHandoffPayload,
            input_builder=finance_input_builder,
            include_input_schema=False,
            max_turns=4,
        ),
        "hr": hr_agent.as_tool(
            tool_name="hr_specialist",
            tool_description="intent + distilled hr data",
            parameters=HRHandoffPayload,
            input_builder=hr_input_builder,
            include_input_schema=False,
            max_turns=3,
        ),
    }


def build_specialist_tools(settings: AppSettings) -> tuple[Any, Any]:
    registry = build_specialist_registry(settings)
    return registry["finance"], registry["hr"]


def _build_hooks():
    agents = import_agents_sdk()

    class DelegationHooks(agents.RunHooks):
        async def on_tool_start(self, context, agent, tool) -> None:
            context.context.delegation_sequence.append(f"{agent.name}:{tool.name}")
            if tool.name == "finance_specialist":
                _consume_context_ttl(context.context.context_ttl_store, "finance")
            elif tool.name == "hr_specialist":
                _consume_context_ttl(context.context.context_ttl_store, "hr")

    return DelegationHooks()


def run_coordination_query(
    settings: AppSettings,
    user_input: str,
    *,
    session_id: str = "coordination-demo",
) -> CoordinationReport:
    agents = import_agents_sdk()
    provider = configure_provider(settings)
    model_router = ModelRouter.from_settings(settings)
    token_budget_manager = TokenBudgetManager(settings.max_context_estimated_tokens)
    observability = ObservabilityManager(
        settings.data_dir,
        "coordination",
        session_id,
    )
    tool_cache = JsonFileCache(
        settings.data_dir / "cache",
        "tool_results",
        settings.tool_cache_ttl_seconds,
    )
    response_cache = JsonFileCache(
        settings.data_dir / "cache",
        "responses",
        settings.response_cache_ttl_seconds,
    )

    memory_manager = StructuredMemoryManager(
        base_dir=settings.data_dir,
        compaction_char_threshold=settings.compaction_char_threshold,
        max_recent_messages=settings.max_recent_messages,
    )
    context = CoordinatorContext(
        finance_service=FinanceService(),
        hr_service=HRService(),
        memory_manager=memory_manager,
        session_id=session_id,
        user_input=user_input,
        tool_cache=tool_cache,
        observability=observability,
        token_budget_manager=token_budget_manager,
        model_routing=model_router.as_dict(),
    )
    contracts = _build_capability_contracts()
    tool_signatures = _build_tool_signatures()
    route_decisions = _route_domains_from_contracts(user_input, contracts)
    handoff_payloads = _build_handoff_payloads(user_input, route_decisions)
    context.route_decisions = route_decisions
    context.distilled_payloads = handoff_payloads
    context.context_ttl_store = _seed_context_ttl_store(
        handoff_payloads,
        default_ttl=settings.context_ttl_steps,
    )
    prompt_segments = _get_cached_prompt_segments(
        route_decisions,
        contracts,
        tool_signatures,
        user_input,
    )
    existing_memory = memory_manager.build_compaction_report(session_id)
    cache_key = stable_cache_key(
        "coordination_response",
        {
            "session_id": session_id,
            "user_input": user_input,
            "route_domains": [decision.domain for decision in route_decisions],
            "handoff_payloads": {
                domain: to_plain_data(payload) for domain, payload in handoff_payloads.items()
            },
            "memory": {
                "case_facts": existing_memory.case_facts,
                "working_memory": existing_memory.working_memory,
            },
            "models": model_router.as_dict(),
        },
    )
    cached = response_cache.get(cache_key)
    if cached.hit:
        observability.record_response_cache_hit()
        observability.trace(
            agent="coordinator",
            tool="response_cache",
            latency_ms=0.0,
            status="cache_hit",
            details={"route_domains": [decision.domain for decision in route_decisions]},
        )
        cached_report = CoordinationReport(**cached.value)
        cached_report.cache_hit = True
        cached_report.telemetry = observability.snapshot()
        cached_report.model_routing = model_router.as_dict()
        return cached_report
    specialist_registry = build_specialist_registry(settings)
    coordinator_tools = [
        specialist_registry[decision.domain] for decision in route_decisions
    ]

    coordinator = agents.Agent(
        name="Coordinator Agent",
        model=model_router.route_for("coordinator"),
        instructions=prompt_segments.system + "\n" + prompt_segments.tools,
        tools=coordinator_tools,
    )

    session = memory_manager.build_sdk_session(
        session_id,
        enable_openai_compaction=provider == "openai",
    )
    prompt_tokens = token_budget_manager.estimate_tokens(
        {
            "instructions": coordinator.instructions,
            "input": _coordinator_execution_input(context),
        }
    )
    observability.record_tokens(prompt_tokens=prompt_tokens)
    started_at = time.perf_counter()
    result = agents.Runner.run_sync(
        coordinator,
        _coordinator_execution_input(context),
        context=context,
        hooks=_build_hooks(),
        session=session,
        run_config=agents.RunConfig(
            call_model_input_filter=_build_coordinator_input_filter(prompt_segments)
        ),
    )
    observability.record_tokens(
        completion_tokens=token_budget_manager.estimate_tokens(str(result.final_output))
    )
    observability.trace(
        agent="coordinator",
        tool="delegate_and_aggregate",
        latency_ms=(time.perf_counter() - started_at) * 1000,
        status="success",
        details={"route_domains": [decision.domain for decision in route_decisions]},
    )
    report = CoordinationReport(
        user_input=user_input,
        final_output=str(result.final_output),
        delegation_sequence=list(context.delegation_sequence),
        payload_log=list(context.payload_log),
        route_domains=[decision.domain for decision in route_decisions],
        capability_contracts=[
            asdict(contracts[decision.domain]) for decision in route_decisions
        ],
        tool_signatures=[
            asdict(tool_signatures[decision.domain]) for decision in route_decisions
        ],
        prompt_segments=asdict(prompt_segments),
        handoff_payloads={
            domain: to_plain_data(payload) for domain, payload in handoff_payloads.items()
        },
        context_ttl_state=_serialize_ttl_state(context.context_ttl_store),
        context_budget=dict(context.last_context_budget_report),
        model_routing=model_router.as_dict(),
        telemetry=observability.snapshot(),
    )
    response_cache.set(cache_key, report)
    return report
