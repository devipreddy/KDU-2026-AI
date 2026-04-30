"""Phase 1: loop detection and circuit breaker workflow."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..runtime.circuit_breaker import CircuitBreaker
from ..runtime.cost_control import JsonFileCache, TokenBudgetManager, stable_cache_key
from ..runtime.loop_control import ToolFailureTracker
from ..runtime.observability import ObservabilityManager
from ..runtime.retry_policy import (
    ExponentialBackoffPolicy,
    RetryBudget,
    classify_failure,
)
from ..schemas import Phase1Report
from ..sdk_support import (
    RunContextWrapper,
    bind_run_context_wrapper,
    configure_provider,
    import_agents_sdk,
)
from ..services.internal_db import InternalDatabaseService
from ..settings import AppSettings

LOGGER = logging.getLogger(__name__)
TOOL_NAME = "query_internal_database"


@dataclass(slots=True)
class Phase1Context:
    service: InternalDatabaseService
    failure_tracker: ToolFailureTracker
    backoff_policy: ExponentialBackoffPolicy
    graceful_fallback: str
    sleep_fn: Callable[[float], None]
    observability: ObservabilityManager | None = None
    tool_cache: JsonFileCache | None = None
    notes: list[str] = field(default_factory=list)


def _phase1_input_filter(data):
    from agents.run import ModelInputData

    context = data.context
    instructions = data.model_data.instructions
    if (
        context is not None
        and not context.failure_tracker.circuit_breaker.allow(TOOL_NAME)
        and instructions is not None
    ):
        instructions += (
            "\nSYSTEM: The circuit breaker for query_internal_database is OPEN. "
            "Do not call the tool again. Return the graceful fallback response."
        )
    if (
        context is not None
        and context.failure_tracker.latest_feedback is not None
        and instructions is not None
    ):
        instructions += (
            "\nSYSTEM RETRY POLICY: "
            + json.dumps(context.failure_tracker.latest_feedback, sort_keys=True)
        )
    return ModelInputData(input=data.model_data.input, instructions=instructions)


def _max_turns_handler(_data):
    from agents import RunErrorHandlerResult

    return RunErrorHandlerResult(
        final_output=(
            "The workflow hit the turn limit after repeated tool failures. "
            "This demonstrates the default loop boundary."
        ),
        include_in_history=False,
    )


def _build_query_tool(guarded: bool):
    agents = import_agents_sdk()
    bind_run_context_wrapper(globals())

    def failure_formatter(ctx: RunContextWrapper[Phase1Context], error: Exception) -> str:
        if not guarded:
            return (
                "TEMPORARY_DATABASE_FAILURE: query_internal_database returned HTTP 500. "
                "Retry only if you still need the answer and the circuit breaker is not open."
            )
        tracker = ctx.context.failure_tracker
        if tracker.latest_feedback is not None:
            return json.dumps(tracker.latest_feedback, sort_keys=True)
        return json.dumps(
            {
                "tool_name": TOOL_NAME,
                "tool_error": str(error),
                "error_type": "unknown_error",
                "retryable": False,
                "retry_allowed": False,
                "instruction": "Do not retry. Provide the graceful fallback.",
                "retry_budget_remaining": tracker.retry_budget.remaining,
                "circuit_breaker_open": not tracker.circuit_breaker.allow(TOOL_NAME),
            },
            sort_keys=True,
        )

    @agents.function_tool(failure_error_function=failure_formatter)
    def query_internal_database(
        ctx: RunContextWrapper[Phase1Context],
        metric: str,
    ) -> str:
        """Query the internal analytics database for company metrics."""

        tracker = ctx.context.failure_tracker
        tracker.record_attempt(TOOL_NAME)
        observability = ctx.context.observability
        started_at = time.perf_counter()

        if guarded and not tracker.circuit_breaker.allow(TOOL_NAME):
            if observability is not None:
                observability.alert(
                    alert_type="circuit_breaker_triggered",
                    severity="warning",
                    message="Circuit breaker blocked a repeated database call.",
                    context={"tool_name": TOOL_NAME},
                )
            return (
                "CIRCUIT_BREAKER_OPEN: database access is blocked after repeated failures. "
                f"Fallback response: {ctx.context.graceful_fallback}"
            )

        cache = ctx.context.tool_cache
        cache_key = stable_cache_key(
            "phase1:active_users",
            {"metric": metric},
        )
        if cache is not None:
            lookup = cache.get(cache_key)
            if lookup.hit:
                if observability is not None:
                    observability.record_tool_cache_hit()
                    observability.trace(
                        agent="analytics",
                        tool=TOOL_NAME,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        status="cache_hit",
                        details={"metric": metric},
                    )
                return str(lookup.value)

        try:
            active_users = ctx.context.service.count_active_users()
        except Exception as exc:
            classification = classify_failure(exc)
            event = tracker.record_failure(
                TOOL_NAME,
                metric,
                classification,
                str(exc),
            )
            if event is not None:
                LOGGER.warning(
                    "Loop detected for %s after %s failures.",
                    TOOL_NAME,
                    event.consecutive_failures,
                )
                if observability is not None:
                    observability.alert(
                        alert_type="loop_detected",
                        severity="warning",
                        message="Repeated failing database calls triggered loop detection.",
                        context={
                            "tool_name": TOOL_NAME,
                            "consecutive_failures": event.consecutive_failures,
                        },
                    )
            if not guarded:
                tracker.latest_feedback = None
                if observability is not None:
                    observability.trace(
                        agent="analytics",
                        tool=TOOL_NAME,
                        latency_ms=(time.perf_counter() - started_at) * 1000,
                        status=classification.error_type,
                        details={"metric": metric, "error": str(exc)},
                    )
                raise

            circuit_open = not tracker.circuit_breaker.allow(TOOL_NAME)
            if classification.max_retry_count is not None:
                attempts = tracker.attempts_for(TOOL_NAME)
                limited_retry_exhausted = attempts >= classification.max_retry_count
            else:
                limited_retry_exhausted = False

            retry_allowed = (
                classification.retryable
                and not circuit_open
                and not limited_retry_exhausted
                and tracker.retry_budget.consume()
            )
            retry_number = max(0, tracker.retry_budget.consumed)
            backoff_seconds = (
                ctx.context.backoff_policy.delay_for(retry_number)
                if retry_allowed
                else 0.0
            )

            if retry_allowed and backoff_seconds > 0:
                LOGGER.info(
                    "Retryable failure for %s. Backing off for %.2fs before allowing another retry.",
                    TOOL_NAME,
                    backoff_seconds,
                )
                if observability is not None:
                    observability.record_retry()
                ctx.context.sleep_fn(backoff_seconds)

            if circuit_open:
                instruction = (
                    "Do NOT retry. The circuit breaker is open. Provide the graceful fallback."
                )
            elif not classification.retryable:
                instruction = (
                    "Do NOT retry. This failure class is non-retryable. Provide the graceful fallback."
                )
            elif limited_retry_exhausted:
                instruction = (
                    "Do NOT retry. Timeout retry limit reached. Provide the graceful fallback."
                )
            elif not retry_allowed:
                instruction = (
                    "Do NOT retry. The global retry budget is exhausted. Provide the graceful fallback."
                )
            else:
                instruction = (
                    "Retry is allowed only if the answer is still required. "
                    "If the next attempt fails again, reassess before retrying."
                )

            tracker.build_feedback(
                tool_name=TOOL_NAME,
                classification=classification,
                error_message=str(exc),
                retry_allowed=retry_allowed,
                backoff_seconds=backoff_seconds,
                instruction=instruction,
            )
            if observability is not None:
                observability.trace(
                    agent="analytics",
                    tool=TOOL_NAME,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status=classification.error_type,
                    details={
                        "metric": metric,
                        "retry_allowed": retry_allowed,
                        "error": str(exc),
                    },
                )
                if circuit_open:
                    observability.alert(
                        alert_type="circuit_breaker_triggered",
                        severity="warning",
                        message="Circuit breaker opened after repeated database failures.",
                        context={"tool_name": TOOL_NAME},
                    )
            raise

        tracker.record_success(TOOL_NAME)
        result = f"Active users: {active_users}"
        if cache is not None:
            cache.set(cache_key, result)
        if observability is not None:
            observability.trace(
                agent="analytics",
                tool=TOOL_NAME,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="success",
                details={"metric": metric},
            )
        return result

    return query_internal_database


def _run_single_variant(
    settings: AppSettings,
    guarded: bool,
    *,
    graceful_fallback: str,
) -> tuple[int, str, ToolFailureTracker, dict[str, object]]:
    agents = import_agents_sdk()
    observability = ObservabilityManager(
        settings.data_dir,
        "phase1-guarded" if guarded else "phase1-baseline",
        "default",
    )
    token_budget_manager = TokenBudgetManager(settings.max_context_estimated_tokens)

    failure_tracker = ToolFailureTracker(
        circuit_breaker=CircuitBreaker(threshold=settings.circuit_breaker_threshold),
        retry_budget=RetryBudget(max_attempts=settings.request_retry_budget),
    )
    context = Phase1Context(
        service=InternalDatabaseService(should_fail=True),
        failure_tracker=failure_tracker,
        backoff_policy=ExponentialBackoffPolicy(
            base_delay_seconds=settings.retry_backoff_base_seconds,
            max_delay_seconds=settings.retry_backoff_max_seconds,
        ),
        graceful_fallback=graceful_fallback,
        sleep_fn=time.sleep,
        observability=observability,
        tool_cache=JsonFileCache(
            settings.data_dir / "cache",
            "tool_results",
            settings.tool_cache_ttl_seconds,
        ),
    )

    instructions = (
        "Count the active users. "
        "Use query_internal_database. "
        "If the tool fails, keep trying until you get a concrete count."
    )
    if guarded:
        instructions += (
            " If the tool or system tells you the circuit breaker is open, "
            "stop retrying and provide the graceful fallback."
        )

    agent = agents.Agent(
        name="Analytics Agent",
        model=settings.phase1_model,
        instructions=instructions,
        tools=[_build_query_tool(guarded=guarded)],
    )
    observability.record_tokens(
        prompt_tokens=token_budget_manager.estimate_tokens(
            {"instructions": instructions, "input": "Count the active users"}
        )
    )

    result = agents.Runner.run_sync(
        agent,
        "Count the active users",
        context=context,
        max_turns=settings.default_max_turns if not guarded else 4,
        error_handlers={"max_turns": _max_turns_handler},
        run_config=agents.RunConfig(
            call_model_input_filter=_phase1_input_filter if guarded else None
        ),
    )
    observability.record_tokens(
        completion_tokens=token_budget_manager.estimate_tokens(str(result.final_output))
    )
    return (
        failure_tracker.attempts_for(TOOL_NAME),
        str(result.final_output),
        failure_tracker,
        observability.snapshot(),
    )


def run_phase1(settings: AppSettings) -> Phase1Report:
    configure_provider(settings)
    graceful_fallback = (
        "I can't access the internal analytics database right now, so I can't confirm "
        "the active user count. Please try again later."
    )
    baseline_attempts, baseline_output, _, baseline_telemetry = _run_single_variant(
        settings,
        guarded=False,
        graceful_fallback=graceful_fallback,
    )
    guarded_attempts, guarded_output, guarded_tracker, guarded_telemetry = _run_single_variant(
        settings,
        guarded=True,
        graceful_fallback=graceful_fallback,
    )

    return Phase1Report(
        baseline_total_attempts=baseline_attempts,
        baseline_retries=max(0, baseline_attempts - 1),
        baseline_final_output=baseline_output,
        guarded_total_attempts=guarded_attempts,
        guarded_retries=max(0, guarded_attempts - 1),
        guarded_final_output=guarded_output,
        circuit_breaker_opened=not guarded_tracker.circuit_breaker.allow(TOOL_NAME),
        loop_events=list(guarded_tracker.loop_events),
        telemetry={
            "baseline": baseline_telemetry,
            "guarded": guarded_telemetry,
        },
    )
