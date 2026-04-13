from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

Currency = Literal["INR", "USD", "EUR"]
ActionType = Literal["BUY", "SELL", "QUOTE", "PORTFOLIO", "HELP", "UNKNOWN"]
ApprovalStatus = Literal["PENDING", "APPROVED", "REJECTED"]
OrderStatus = Literal["PENDING_APPROVAL", "EXECUTED", "REJECTED", "FAILED"]
MessageRole = Literal["user", "assistant", "system"]
SUPPORTED_CURRENCIES = {"INR", "USD", "EUR"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class AgentInput(BaseModel):
    thread_id: str
    user_id: str
    content: str
    base_currency: Currency = "INR"

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content cannot be empty")
        return cleaned


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class OrderIntent(BaseModel):
    action: ActionType = "UNKNOWN"
    symbol: str | None = None
    quantity: float | None = None
    confidence: float = 1.0
    notes: list[str] = Field(default_factory=list)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class UserProfile(BaseModel):
    user_id: str = ""
    base_currency: Currency = "INR"


class Holding(BaseModel):
    symbol: str
    quantity: float
    average_cost_base: float
    price_native: float
    currency_native: Currency
    price_base: float
    market_value_base: float
    updated_at: datetime = Field(default_factory=utc_now)


class Portfolio(BaseModel):
    cash_balance_base: float = 1_000_000.0
    holdings: dict[str, Holding] = Field(default_factory=dict)
    total_market_value_base: float = 0.0
    total_equity_base: float = 1_000_000.0


class MarketQuote(BaseModel):
    symbol: str
    price_native: float
    currency_native: Currency
    price_base: float | None = None
    base_currency: Currency | None = None
    fx_rate: float = 1.0
    as_of: datetime = Field(default_factory=utc_now)
    provider: str = "mock"


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: new_id("ord"))
    action: Literal["BUY", "SELL"]
    symbol: str
    quantity: float
    status: OrderStatus
    price_base: float
    notional_base: float
    currency_base: Currency
    created_at: datetime = Field(default_factory=utc_now)
    approval_id: str | None = None


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: new_id("txn"))
    order_id: str
    action: Literal["BUY", "SELL"]
    symbol: str
    quantity: float
    amount_base: float
    currency_base: Currency
    created_at: datetime = Field(default_factory=utc_now)


class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("apr"))
    action: Literal["BUY", "SELL"]
    symbol: str
    quantity: float
    estimated_notional_base: float
    base_currency: Currency
    status: ApprovalStatus = "PENDING"
    requested_at: datetime = Field(default_factory=utc_now)


class ApprovalDecision(BaseModel):
    approval_id: str
    approved: bool
    reviewer: str = "human-reviewer"
    reason: str | None = None
    decided_at: datetime = Field(default_factory=utc_now)


class UsageRecord(BaseModel):
    source: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionEvent(BaseModel):
    node_name: str
    decision_type: str
    selected_route: str
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ToolEvent(BaseModel):
    tool_name: str
    succeeded: bool
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    error: str | None = None


class TurnObservation(BaseModel):
    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    thread_id: str
    user_message: str
    assistant_response: str
    route_path: list[str] = Field(default_factory=list)
    decision_count: int = 0
    tool_call_count: int = 0
    interrupted: bool = False
    approval_status: ApprovalStatus | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)


class ExecutionState(BaseModel):
    session_id: str = ""
    thread_id: str = ""
    current_node: str = "start"
    next_node_hint: str | None = None
    step_count: int = 0
    retry_count: int = 0
    route_history: list[str] = Field(default_factory=list)
    current_turn_routes: list[str] = Field(default_factory=list)
    current_turn_id: str = ""
    turn_started_at: datetime = Field(default_factory=utc_now)
    turn_start_input_tokens: int = 0
    turn_start_output_tokens: int = 0
    turn_start_total_tokens: int = 0
    turn_start_estimated_cost_usd: float = 0.0
    turn_start_tool_event_count: int = 0
    turn_start_decision_event_count: int = 0
    last_error: str | None = None
    last_transition_at: datetime = Field(default_factory=utc_now)


class ConversationState(BaseModel):
    messages: list[ConversationMessage] = Field(default_factory=list)
    context_summary: str = ""
    intent: OrderIntent | None = None


class DomainState(BaseModel):
    user: UserProfile = Field(default_factory=UserProfile)
    portfolio: Portfolio = Field(default_factory=Portfolio)
    orders: list[Order] = Field(default_factory=list)
    ledger: list[Transaction] = Field(default_factory=list)
    latest_quote: MarketQuote | None = None


class ControlState(BaseModel):
    pending_approval: ApprovalRequest | None = None
    last_decision: ApprovalDecision | None = None
    flags: list[str] = Field(default_factory=list)
    blocked: bool = False
    requires_human_input: bool = False


class TelemetryState(BaseModel):
    decision_events: list[DecisionEvent] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    usage: list[UsageRecord] = Field(default_factory=list)
    turns: list[TurnObservation] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class InteractionAnalytics(BaseModel):
    route_path: list[str] = Field(default_factory=list)
    decision_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    approval_pending: bool = False
    market_data_provider: str
    intent_parser_mode: str
    langsmith_tracing_enabled: bool = False
    langsmith_project: str = ""


def ensure_supported_currency(value: str) -> Currency:
    code = value.upper()
    if code not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency '{value}'. Supported currencies: {sorted(SUPPORTED_CURRENCIES)}")
    return code  # type: ignore[return-value]
