from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Complexity = Literal["low", "medium", "high"]
ResponseType = Literal["simple", "standard", "complex"]


class AppSettings(BaseModel):
    name: str
    environment: str = "local"
    timezone: str = "UTC"
    state_db_path: str = "state/fixit_llmops.sqlite3"
    log_level: str = "INFO"


class LLMProviderConfig(BaseModel):
    provider: str = "openrouter-compatible"
    base_url: str
    api_key_env: str
    timeout_seconds: float = 30.0
    max_retries: int = 1
    default_headers: dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False


class ModelSpec(BaseModel):
    model_id: str
    max_output_tokens: int = 256
    temperature: float = 0.2
    input_cost_per_1m: float
    output_cost_per_1m: float


class BudgetConfig(BaseModel):
    monthly_limit_usd: float
    daily_limit_usd: float | None = None
    warn_ratio: float = 0.8
    degrade_ratio: float = 0.9
    hard_stop_ratio: float = 1.0

    @model_validator(mode="after")
    def set_daily_limit(self) -> "BudgetConfig":
        if self.daily_limit_usd is None:
            self.daily_limit_usd = round(self.monthly_limit_usd / 30.0, 2)
        return self


class FeatureFlags(BaseModel):
    enable_budget_guardrails: bool = True
    enable_hot_reload: bool = True
    enable_request_logging: bool = True
    enable_llm_classifier_fallback: bool = False
    enable_model_fallbacks: bool = True
    enable_safe_fallback: bool = True


class PromptConfig(BaseModel):
    root_dir: str = "prompts"
    selection_strategy: Literal["latest_stable", "best_performing"] = "latest_stable"
    fallback_category: str = "fallback"


class CategoryRule(BaseModel):
    response_type: ResponseType
    default_complexity: Complexity
    keywords: list[str] = Field(default_factory=list)
    regexes: list[str] = Field(default_factory=list)


class ClassificationConfig(BaseModel):
    fallback_category: str = "fallback"
    short_query_word_threshold: int = 10
    long_query_word_threshold: int = 18
    very_long_query_word_threshold: int = 32
    escalation_keywords: list[str] = Field(default_factory=list)
    medium_keywords: list[str] = Field(default_factory=list)
    categories: dict[str, CategoryRule]


class RoutingConfig(BaseModel):
    default_model_alias: str = "low"
    complexity_to_model: dict[str, str]
    fallback_order: dict[str, list[str]] = Field(default_factory=dict)


class AnalysisAverageTokens(BaseModel):
    prompt: int
    completion: int


class AnalysisConfig(BaseModel):
    monthly_queries: int = 300000
    legacy_all_high_cost_usd: float = 3000.0
    projected_query_mix: dict[str, float] = Field(default_factory=lambda: {"low": 0.85, "high": 0.15})
    average_tokens: dict[str, AnalysisAverageTokens] = Field(default_factory=dict)


class FixItConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    app: AppSettings
    llm: LLMProviderConfig
    models: dict[str, ModelSpec]
    budget: BudgetConfig
    feature_flags: FeatureFlags
    prompts: PromptConfig
    classification: ClassificationConfig
    routing: RoutingConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)


class PromptAsset(BaseModel):
    prompt_id: str
    category: str
    version: str
    status: Literal["stable", "experimental", "archived"]
    selection_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    template: str


class ClassificationResult(BaseModel):
    category: str
    complexity: Complexity
    response_type: ResponseType
    confidence: float
    matched_rules: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMGeneration(BaseModel):
    text: str
    model_alias: str
    model_id: str
    usage: TokenUsage
    raw_response_id: str | None = None


class BudgetDecision(BaseModel):
    selected_model_alias: str | None
    action: str
    reason: str | None = None
    daily_spend_usd: float = 0.0
    monthly_spend_usd: float = 0.0
    daily_remaining_usd: float = 0.0
    monthly_remaining_usd: float = 0.0


class SupportRequest(BaseModel):
    query: str
    customer_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class SupportResponse(BaseModel):
    request_id: str
    reply: str
    category: str
    complexity: Complexity
    response_type: ResponseType
    selected_model_alias: str | None
    selected_model_id: str | None
    prompt_id: str
    prompt_version: str
    estimated_cost_usd: float
    budget_action: str
    fallback_reason: str | None = None
    latency_ms: float


class RequestLogRecord(BaseModel):
    request_id: str
    timestamp: datetime
    query: str
    category: str
    complexity: Complexity
    response_type: ResponseType
    requested_model_alias: str
    used_model_alias: str | None
    model_id: str | None
    prompt_id: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    budget_action: str
    classification_confidence: float
    response_preview: str

