"""Phase 5 planner-executor workflow."""

from __future__ import annotations

import json
import time
from typing import Any

from ..memory.manager import StructuredMemoryManager
from ..runtime.cost_control import (
    JsonFileCache,
    ModelRouter,
    TokenBudgetManager,
    stable_cache_key,
)
from ..runtime.observability import ObservabilityManager
from ..schemas import (
    ExecutionPlan,
    PlanValidationReport,
    PlannerExecutorReport,
    PlannerMemoryPacket,
    PlanStep,
    StepExecutionResult,
    to_plain_data,
)
from ..sdk_support import configure_provider, import_agents_sdk
from ..services.finance import FinanceService
from ..services.hr import HRService
from ..settings import AppSettings
from .orchestration import CoordinatorContext, build_specialist_registry

ACTION_CATALOG = {
    "finance_specialist": {
        "get_salary": {"required_fields": ("employee_name",)},
        "update_bank_details": {"required_fields": ("routing_number",)},
    },
    "hr_specialist": {
        "get_pto_balance": {"required_fields": ("employee_name",)},
    },
}

SUCCESS_STATUSES = {"completed", "done", "success"}
NON_BLOCKING_STATUSES = SUCCESS_STATUSES | {"requires_user_input"}


def _flatten_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "data" and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[nested_key] = nested_value
            continue
        flattened[key] = value
    return flattened


def _memory_fields_for_target(
    packet: PlannerMemoryPacket,
    agent: str,
) -> dict[str, Any]:
    target_memory = _memory_for_target(packet, agent)
    working_memory = target_memory.get("working_memory", {})
    case_facts = target_memory.get("case_facts", {})
    available = dict(working_memory.get("active_entities", {}))
    banking_update = case_facts.get("banking_update", {})
    if isinstance(banking_update, dict):
        for key, value in banking_update.items():
            if value is not None:
                available[key] = value
    return available


def _build_plan_validation_report(
    plan: ExecutionPlan,
    packet: PlannerMemoryPacket | None = None,
) -> PlanValidationReport:
    report = PlanValidationReport(is_valid=True)
    seen_ids: set[str] = set()
    all_actions = {
        action
        for actions_by_agent in ACTION_CATALOG.values()
        for action in actions_by_agent
    }
    memory_fields_by_agent = {
        agent: _memory_fields_for_target(packet, agent) if packet is not None else {}
        for agent in ACTION_CATALOG
    }

    for step in plan.steps:
        if step.step_id in seen_ids:
            report.duplicate_step_ids.append(step.step_id)
            continue
        seen_ids.add(step.step_id)
        report.validated_steps.append(step.step_id)

        if step.agent not in ACTION_CATALOG:
            report.agent_mismatches[step.step_id] = (
                f"Unsupported agent: {step.agent}"
            )
            continue

        if step.action not in all_actions:
            report.invalid_actions[step.step_id] = (
                f"Unsupported action: {step.action}"
            )
            continue

        if step.action not in ACTION_CATALOG[step.agent]:
            report.agent_mismatches[step.step_id] = (
                f"Action {step.action} is not allowed for {step.agent}."
            )
            continue

        payload_fields = _flatten_payload_fields(step.payload)
        if payload_fields.get("intent") and payload_fields["intent"] != step.action:
            report.invalid_actions[step.step_id] = (
                "Payload intent does not match step action."
            )
            continue

        required_fields = ACTION_CATALOG[step.agent][step.action]["required_fields"]
        available_fields = dict(memory_fields_by_agent[step.agent])
        available_fields.update(payload_fields)
        missing = [
            field_name
            for field_name in required_fields
            if not available_fields.get(field_name)
        ]
        if missing:
            report.missing_fields[step.step_id] = missing

    known_steps = {step.step_id for step in plan.steps}
    for step in plan.steps:
        invalid_dependencies = []
        for dependency in step.depends_on:
            if dependency == step.step_id:
                invalid_dependencies.append(dependency)
            elif dependency not in known_steps:
                invalid_dependencies.append(dependency)
        if invalid_dependencies:
            report.invalid_dependencies[step.step_id] = invalid_dependencies

    report.is_valid = not any(
        (
            report.duplicate_step_ids,
            report.invalid_actions,
            report.agent_mismatches,
            report.missing_fields,
            report.invalid_dependencies,
        )
    )
    return report


def _raise_for_validation_errors(report: PlanValidationReport) -> None:
    if report.is_valid:
        return

    error_parts: list[str] = []
    if report.duplicate_step_ids:
        error_parts.append(
            f"duplicate_step_ids={sorted(report.duplicate_step_ids)}"
        )
    if report.invalid_actions:
        error_parts.append(f"invalid_actions={report.invalid_actions}")
    if report.agent_mismatches:
        error_parts.append(f"agent_mismatches={report.agent_mismatches}")
    if report.missing_fields:
        error_parts.append(f"missing_fields={report.missing_fields}")
    if report.invalid_dependencies:
        error_parts.append(f"invalid_dependencies={report.invalid_dependencies}")
    raise ValueError("Plan validation failed: " + "; ".join(error_parts))


def _validate_plan(
    plan: ExecutionPlan,
    packet: PlannerMemoryPacket | None = None,
) -> ExecutionPlan:
    report = _build_plan_validation_report(plan, packet)
    _raise_for_validation_errors(report)
    return plan


def _topological_batches(plan: ExecutionPlan) -> list[list[PlanStep]]:
    batches: list[list[PlanStep]] = []
    completed: set[str] = set()
    remaining = {step.step_id: step for step in plan.steps}
    while remaining:
        ready = [
            step
            for step in remaining.values()
            if all(dependency in completed for dependency in step.depends_on)
        ]
        if not ready:
            raise ValueError("Plan contains unresolved or cyclic dependencies.")
        ready.sort(key=lambda step: step.step_id)
        batches.append(list(ready))
        for step in ready:
            completed.add(step.step_id)
            remaining.pop(step.step_id, None)
    return batches


def _topological_steps(plan: ExecutionPlan) -> list[PlanStep]:
    ordered: list[PlanStep] = []
    for batch in _topological_batches(plan):
        ordered.extend(batch)
    return ordered


def _build_memory_packet(
    memory_manager: StructuredMemoryManager,
    session_id: str,
) -> PlannerMemoryPacket:
    report = memory_manager.build_compaction_report(session_id)
    return PlannerMemoryPacket(
        session_id=session_id,
        short_term_window=list(report.short_term_window),
        working_memory=dict(report.working_memory),
        case_facts=dict(report.case_facts),
    )


def _planner_memory_packet_from_plain(payload: dict[str, Any]) -> PlannerMemoryPacket:
    return PlannerMemoryPacket(
        session_id=payload.get("session_id", ""),
        short_term_window=list(payload.get("short_term_window", [])),
        working_memory=dict(payload.get("working_memory", {})),
        case_facts=dict(payload.get("case_facts", {})),
    )


def _compress_memory_packet(
    packet: PlannerMemoryPacket,
    token_budget_manager: TokenBudgetManager,
) -> tuple[PlannerMemoryPacket, dict[str, Any]]:
    compressed_payload, budget_report = token_budget_manager.compress_context(
        to_plain_data(packet)
    )
    return _planner_memory_packet_from_plain(compressed_payload), budget_report.as_dict()


def _memory_for_target(
    packet: PlannerMemoryPacket,
    target: str,
) -> dict[str, Any]:
    case_facts = packet.case_facts
    working_memory = packet.working_memory
    if target == "finance_specialist":
        relevant_case_facts = {
            "financials": case_facts.get("financials", {}),
            "orders": case_facts.get("orders", {}),
            "transactions": case_facts.get("transactions", {}),
            "banking_update": case_facts.get("banking_update", {}),
            "requires_user_input": case_facts.get("requires_user_input", False),
            "missing_fields": case_facts.get("missing_fields", []),
        }
    elif target == "hr_specialist":
        relevant_case_facts = {
            "recent_relevant_messages": case_facts.get("recent_relevant_messages", []),
            "final_decisions": case_facts.get("final_decisions", []),
            "requires_user_input": case_facts.get("requires_user_input", False),
            "missing_fields": case_facts.get("missing_fields", []),
        }
    else:
        relevant_case_facts = case_facts

    relevant_working_memory = {
        "current_task": working_memory.get("current_task"),
        "active_entities": working_memory.get("active_entities", {}),
        "pending_questions": working_memory.get("pending_questions", []),
        "recent_decisions": working_memory.get("recent_decisions", []),
    }
    return {
        "session_id": packet.session_id,
        "short_term_window": packet.short_term_window,
        "working_memory": relevant_working_memory,
        "case_facts": relevant_case_facts,
    }


def _planner_prompt(request: str, packet: PlannerMemoryPacket) -> str:
    return (
        f"User request: {request}\n"
        "Memory packet:\n"
        f"{json.dumps(to_plain_data(packet), indent=2, sort_keys=True)}\n"
        "Return an ExecutionPlan.\n"
        "Rules:\n"
        "- Each step must include step_id, action, agent, description, payload, and depends_on.\n"
        "- Only use agents finance_specialist and hr_specialist.\n"
        "- Allowed actions: finance_specialist -> get_salary, update_bank_details; hr_specialist -> get_pto_balance.\n"
        "- Payloads must stay small and structured.\n"
        "- Finance payloads should look like {intent, data{employee_name?, routing_number?, account_number?, account_holder_name?}}.\n"
        "- HR payloads should look like {intent, data{employee_name?}}.\n"
        "- Keep dependencies explicit and leave independent steps without unnecessary dependencies."
    )


def _step_prompt(
    *,
    request: str,
    plan: ExecutionPlan,
    step: PlanStep,
    target_memory: dict[str, Any],
    execution_ledger: dict[str, Any],
) -> str:
    return (
        f"User request: {request}\n"
        f"Plan goal: {plan.goal}\n"
        f"Plan assumptions: {json.dumps(plan.assumptions, sort_keys=True)}\n"
        f"Plan success criteria: {json.dumps(plan.success_criteria, sort_keys=True)}\n"
        "Execute exactly this step:\n"
        f"{json.dumps(to_plain_data(step), indent=2, sort_keys=True)}\n"
        "Relevant memory packet:\n"
        f"{json.dumps(target_memory, indent=2, sort_keys=True)}\n"
        "Execution ledger so far:\n"
        f"{json.dumps(execution_ledger, indent=2, sort_keys=True)}\n"
        "Return StepExecutionResult with status set to one of completed, failed, or requires_user_input. "
        "Do not re-plan. Use the available tool if needed."
    )


def _ledger_from_results(results: list[StepExecutionResult]) -> dict[str, Any]:
    return {
        result.step_id: {
            "status": result.status,
            "result_summary": result.result_summary,
            "memory_updates": result.memory_updates,
        }
        for result in results
    }


def _postprocess_step_result(
    step: PlanStep,
    result: StepExecutionResult,
    packet: PlannerMemoryPacket,
    memory_snapshot_used: dict[str, Any] | None = None,
) -> StepExecutionResult:
    if not result.memory_updates:
        result.memory_updates = {
            "target": step.target,
            "step_payload": to_plain_data(step.payload),
            "memory_snapshot_used": memory_snapshot_used
            if memory_snapshot_used is not None
            else _memory_for_target(packet, step.target),
        }
    return result


def _is_successful_status(status: str) -> bool:
    return status.strip().lower() in SUCCESS_STATUSES


def _normalize_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in SUCCESS_STATUSES:
        return "completed"
    if normalized == "requires_user_input":
        return normalized
    return "failed"


def _dependencies_are_completed(
    step: PlanStep,
    execution_state: dict[str, str],
) -> bool:
    return all(execution_state.get(dependency) == "completed" for dependency in step.depends_on)


def _plan_step_from_plain(payload: dict[str, Any]) -> PlanStep:
    return PlanStep(
        step_id=payload["step_id"],
        action=payload["action"],
        agent=payload["agent"],
        description=payload.get("description", ""),
        payload=dict(payload.get("payload", {})),
        depends_on=list(payload.get("depends_on", [])),
    )


def _execution_plan_from_plain(payload: dict[str, Any]) -> ExecutionPlan:
    return ExecutionPlan(
        goal=payload["goal"],
        assumptions=list(payload.get("assumptions", [])),
        steps=[_plan_step_from_plain(step) for step in payload.get("steps", [])],
        success_criteria=list(payload.get("success_criteria", [])),
    )


def _step_result_from_plain(payload: dict[str, Any]) -> StepExecutionResult:
    return StepExecutionResult(
        step_id=payload["step_id"],
        status=payload["status"],
        result_summary=payload["result_summary"],
        memory_updates=dict(payload.get("memory_updates", {})),
    )


def _planner_report_from_plain(payload: dict[str, Any]) -> PlannerExecutorReport:
    return PlannerExecutorReport(
        request=payload["request"],
        plan=_execution_plan_from_plain(payload["plan"]),
        results=[_step_result_from_plain(item) for item in payload.get("results", [])],
        final_output=payload.get("final_output", ""),
        cache_hit=payload.get("cache_hit", False),
        plan_validation=dict(payload.get("plan_validation", {})),
        planner_memory_packet=dict(payload.get("planner_memory_packet", {})),
        executor_memory_packets=dict(payload.get("executor_memory_packets", {})),
        execution_batches=[list(batch) for batch in payload.get("execution_batches", [])],
        execution_ledger=dict(payload.get("execution_ledger", {})),
        execution_state=dict(payload.get("execution_state", {})),
        context_budget=dict(payload.get("context_budget", {})),
        model_routing=dict(payload.get("model_routing", {})),
        telemetry=dict(payload.get("telemetry", {})),
    )


def run_planner_executor(
    settings: AppSettings,
    request: str,
    *,
    session_id: str = "planner-executor-demo",
) -> PlannerExecutorReport:
    agents = import_agents_sdk()
    configure_provider(settings)
    model_router = ModelRouter.from_settings(settings)
    token_budget_manager = TokenBudgetManager(settings.max_context_estimated_tokens)
    observability = ObservabilityManager(
        settings.data_dir,
        "planner-executor",
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
    planner_memory_packet = _build_memory_packet(memory_manager, session_id)
    planner_memory_packet, planner_budget = _compress_memory_packet(
        planner_memory_packet,
        token_budget_manager,
    )
    cache_key = stable_cache_key(
        "planner_executor_response",
        {
            "session_id": session_id,
            "request": request,
            "planner_memory_packet": to_plain_data(planner_memory_packet),
            "models": model_router.as_dict(),
        },
    )
    cached = response_cache.get(cache_key)
    if cached.hit:
        observability.record_response_cache_hit()
        observability.trace(
            agent="planner_executor",
            tool="response_cache",
            latency_ms=0.0,
            status="cache_hit",
            details={"session_id": session_id},
        )
        cached_report = _planner_report_from_plain(cached.value)
        cached_report.cache_hit = True
        cached_report.model_routing = model_router.as_dict()
        cached_report.telemetry = observability.snapshot()
        return cached_report

    planner = agents.Agent(
        name="Planner Agent",
        model=model_router.route_for("planning"),
        instructions=(
            "Create a concise execution plan as structured JSON. "
            "Only use the targets finance_specialist and hr_specialist. "
            "Split independent work into separate steps, and declare dependencies explicitly."
        ),
        output_type=agents.AgentOutputSchema(
            ExecutionPlan,
            strict_json_schema=False,
        ),
    )
    planner_prompt = _planner_prompt(request, planner_memory_packet)
    observability.record_tokens(
        prompt_tokens=token_budget_manager.estimate_tokens(
            {"instructions": planner.instructions, "input": planner_prompt}
        )
    )
    planner_started_at = time.perf_counter()
    planning_result = agents.Runner.run_sync(
        planner,
        planner_prompt,
    )
    observability.record_tokens(
        completion_tokens=token_budget_manager.estimate_tokens(
            to_plain_data(planning_result.final_output)
        )
    )
    observability.trace(
        agent="planner",
        tool="generate_execution_plan",
        latency_ms=(time.perf_counter() - planner_started_at) * 1000,
        status="success",
        details={"session_id": session_id},
    )
    plan = planning_result.final_output
    plan_validation = _build_plan_validation_report(plan, planner_memory_packet)
    if not plan_validation.is_valid:
        observability.alert(
            alert_type="plan_validation_failed",
            severity="warning",
            message="Planner produced an invalid execution plan.",
            context=to_plain_data(plan_validation),
        )
    _raise_for_validation_errors(plan_validation)

    specialist_registry = build_specialist_registry(settings)
    execution_context = CoordinatorContext(
        finance_service=FinanceService(),
        hr_service=HRService(),
        memory_manager=memory_manager,
        session_id=session_id,
        user_input=request,
        tool_cache=tool_cache,
        observability=observability,
        token_budget_manager=token_budget_manager,
        model_routing=model_router.as_dict(),
    )
    results: list[StepExecutionResult] = []
    executor_memory_packets: dict[str, dict[str, Any]] = {}
    execution_batches = _topological_batches(plan)
    execution_state = {step.step_id: "pending" for step in plan.steps}
    executor_budget_reports: dict[str, dict[str, Any]] = {}

    for batch in execution_batches:
        for step in batch:
            if not _dependencies_are_completed(step, execution_state):
                execution_state[step.step_id] = "blocked"
                observability.alert(
                    alert_type="dependency_blocked",
                    severity="warning",
                    message="A plan step was blocked because its dependencies did not complete successfully.",
                    context={"step_id": step.step_id, "depends_on": list(step.depends_on)},
                )
                continue

            current_memory_packet = _build_memory_packet(memory_manager, session_id)
            current_memory_packet, _ = _compress_memory_packet(
                current_memory_packet,
                token_budget_manager,
            )
            target_memory = _memory_for_target(current_memory_packet, step.target)
            target_memory, step_budget = token_budget_manager.compress_context(target_memory)
            executor_budget_reports[step.step_id] = step_budget.as_dict()
            execution_ledger = _ledger_from_results(results)
            execution_state[step.step_id] = "running"
            executor_memory_packets[step.step_id] = {
                "agent": step.agent,
                "action": step.action,
                "memory_packet": target_memory,
                "execution_ledger_before_step": execution_ledger,
                "context_budget": step_budget.as_dict(),
            }
            executor = agents.Agent(
                name="Executor Agent",
                model=model_router.route_for("execution"),
                instructions=(
                    "You execute exactly one validated step at a time. "
                    "Do not re-plan. Use the single available tool when the step requires external execution. "
                    "Return a StepExecutionResult with a concise result_summary and any memory_updates."
                ),
                tools=[specialist_registry[step.target.removesuffix("_specialist")]],
                output_type=agents.AgentOutputSchema(
                    StepExecutionResult,
                    strict_json_schema=False,
                ),
            )
            step_prompt = _step_prompt(
                request=request,
                plan=plan,
                step=step,
                target_memory=target_memory,
                execution_ledger=execution_ledger,
            )
            observability.record_tokens(
                prompt_tokens=token_budget_manager.estimate_tokens(
                    {"instructions": executor.instructions, "input": step_prompt}
                )
            )
            executor_started_at = time.perf_counter()
            execution_result = agents.Runner.run_sync(
                executor,
                step_prompt,
                context=execution_context,
            )
            step_result = _postprocess_step_result(
                step,
                execution_result.final_output,
                current_memory_packet,
                target_memory,
            )
            step_result.status = _normalize_status(step_result.status)
            observability.record_tokens(
                completion_tokens=token_budget_manager.estimate_tokens(
                    to_plain_data(step_result)
                )
            )
            results.append(step_result)
            execution_state[step.step_id] = step_result.status
            memory_manager.ingest_message(session_id, "assistant", step_result.result_summary)
            observability.trace(
                agent="executor",
                tool=step.action,
                latency_ms=(time.perf_counter() - executor_started_at) * 1000,
                status=step_result.status,
                details={"step_id": step.step_id, "agent": step.agent},
            )
            if step_result.status == "requires_user_input":
                observability.alert(
                    alert_type="missing_required_fields",
                    severity="warning",
                    message="Executor step requires additional user input before continuing.",
                    context={"step_id": step.step_id, "memory_updates": step_result.memory_updates},
                )

    final_output = "\n".join(
        f"{result.step_id}: {result.result_summary}" for result in results
    )
    report = PlannerExecutorReport(
        request=request,
        plan=plan,
        results=results,
        final_output=final_output,
        plan_validation=to_plain_data(plan_validation),
        planner_memory_packet=to_plain_data(planner_memory_packet),
        executor_memory_packets=executor_memory_packets,
        execution_batches=[[step.step_id for step in batch] for batch in execution_batches],
        execution_ledger=_ledger_from_results(results),
        execution_state=execution_state,
        context_budget={
            "planner": planner_budget,
            "executors": executor_budget_reports,
        },
        model_routing=model_router.as_dict(),
        telemetry=observability.snapshot(),
    )
    response_cache.set(cache_key, report)
    return report
