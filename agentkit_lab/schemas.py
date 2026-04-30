"""Shared dataclasses for workflow state, memory, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


@dataclass(slots=True)
class LoopDetectionEvent:
    tool_name: str
    consecutive_failures: int
    argument_fingerprint: str
    reason: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class Phase1Report:
    baseline_total_attempts: int
    baseline_retries: int
    baseline_final_output: str
    guarded_total_attempts: int
    guarded_retries: int
    guarded_final_output: str
    circuit_breaker_opened: bool
    loop_events: list[LoopDetectionEvent] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinanceRequest:
    intent: str
    employee_name: str | None = None
    routing_number: str | None = None
    account_number: str | None = None
    account_holder_name: str | None = None


@dataclass(slots=True)
class HRRequest:
    intent: str
    employee_name: str


@dataclass(slots=True)
class FinanceHandoffData:
    employee_name: str | None = None
    routing_number: str | None = None
    account_number: str | None = None
    account_holder_name: str | None = None


@dataclass(slots=True)
class FinanceHandoffPayload:
    intent: str
    data: FinanceHandoffData = field(default_factory=FinanceHandoffData)


@dataclass(slots=True)
class HRHandoffData:
    employee_name: str | None = None


@dataclass(slots=True)
class HRHandoffPayload:
    intent: str
    data: HRHandoffData = field(default_factory=HRHandoffData)


@dataclass(slots=True)
class ContextTTLItem:
    value: Any
    ttl: int


@dataclass(slots=True)
class CoordinationReport:
    user_input: str
    final_output: str
    cache_hit: bool = False
    delegation_sequence: list[str] = field(default_factory=list)
    payload_log: list[dict[str, Any]] = field(default_factory=list)
    route_domains: list[str] = field(default_factory=list)
    capability_contracts: list[dict[str, Any]] = field(default_factory=list)
    tool_signatures: list[dict[str, Any]] = field(default_factory=list)
    prompt_segments: dict[str, str] = field(default_factory=dict)
    handoff_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_ttl_state: dict[str, dict[str, int]] = field(default_factory=dict)
    context_budget: dict[str, Any] = field(default_factory=dict)
    model_routing: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CapabilityContract:
    agent: str
    can_do: list[str]
    cannot_do: list[str]
    delegation_tool: str


@dataclass(slots=True)
class CompressedToolSignature:
    name: str
    args: list[str]
    summary: str


@dataclass(slots=True)
class PromptSegments:
    system: str
    tools: str
    context: str
    user: str


@dataclass(slots=True)
class RoutingDecision:
    domain: str
    matched_capabilities: list[str] = field(default_factory=list)
    matched_positions: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class TransactionFact:
    fact_id: str
    order_id: str | None = None
    transaction_id: str | None = None
    amount: str | None = None
    currency: str | None = None
    status: str | None = None
    raw_excerpt: str | None = None


@dataclass(slots=True)
class OrderFact:
    id: str
    amount: str | None = None
    currency: str | None = None
    transaction_id: str | None = None
    status: str | None = None
    raw_excerpt: str | None = None


@dataclass(slots=True)
class FinancialSnapshot:
    captured_amounts: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    latest_balance: str | None = None


@dataclass(slots=True)
class BankingUpdateFact:
    update_type: str = "bank_account"
    routing_number: str | None = None
    account_number: str | None = None
    account_holder_name: str | None = None
    card_last4: str | None = None
    cvv: str | None = None


@dataclass(slots=True)
class WorkingMemoryState:
    session_id: str
    current_task: str | None = None
    active_entities: dict[str, str] = field(default_factory=dict)
    incremental_summary: str = ""
    recent_decisions: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    last_compacted_message_index: int = 0
    extraction_strategy: str = "cheap_deterministic_extractor"
    reasoning_strategy: str = "reasoning_agent"


@dataclass(slots=True)
class CaseFacts:
    session_id: str
    summary: str = ""
    numerical_data: dict[str, list[str]] = field(default_factory=dict)
    financials: FinancialSnapshot = field(default_factory=FinancialSnapshot)
    orders: dict[str, OrderFact] = field(default_factory=dict)
    transactions: dict[str, TransactionFact] = field(default_factory=dict)
    banking_update: BankingUpdateFact = field(default_factory=BankingUpdateFact)
    final_decisions: list[str] = field(default_factory=list)
    recent_relevant_messages: list[str] = field(default_factory=list)
    requires_user_input: bool = False
    missing_fields: list[str] = field(default_factory=list)
    last_compacted_at: str | None = None


@dataclass(slots=True)
class MemoryCompactionReport:
    session_id: str
    retained_message_count: int
    summary: str
    numerical_keys: list[str]
    transaction_ids: list[str]
    requires_user_input: bool
    missing_fields: list[str]
    case_facts: dict[str, Any] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    short_term_window: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlanStep:
    step_id: str
    action: str
    agent: str
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return self.agent


@dataclass(slots=True)
class ExecutionPlan:
    goal: str
    assumptions: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StepExecutionResult:
    step_id: str
    status: str
    result_summary: str
    memory_updates: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlanValidationReport:
    is_valid: bool
    validated_steps: list[str] = field(default_factory=list)
    duplicate_step_ids: list[str] = field(default_factory=list)
    invalid_actions: dict[str, str] = field(default_factory=dict)
    agent_mismatches: dict[str, str] = field(default_factory=dict)
    missing_fields: dict[str, list[str]] = field(default_factory=dict)
    invalid_dependencies: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerMemoryPacket:
    session_id: str
    short_term_window: list[str] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    case_facts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerExecutorReport:
    request: str
    plan: ExecutionPlan
    results: list[StepExecutionResult] = field(default_factory=list)
    final_output: str = ""
    cache_hit: bool = False
    plan_validation: dict[str, Any] = field(default_factory=dict)
    planner_memory_packet: dict[str, Any] = field(default_factory=dict)
    executor_memory_packets: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_batches: list[list[str]] = field(default_factory=list)
    execution_ledger: dict[str, Any] = field(default_factory=dict)
    execution_state: dict[str, str] = field(default_factory=dict)
    context_budget: dict[str, Any] = field(default_factory=dict)
    model_routing: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
