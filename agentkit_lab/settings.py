"""Environment-driven application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    """Runtime configuration for all phases."""

    project_root: Path
    data_dir: Path
    openai_api_key: str | None
    openrouter_api_key: str | None
    openrouter_base_url: str
    default_max_turns: int = 10
    circuit_breaker_threshold: int = 3
    request_retry_budget: int = 5
    retry_backoff_base_seconds: float = 0.25
    retry_backoff_max_seconds: float = 2.0
    context_ttl_steps: int = 2
    max_context_estimated_tokens: int = 700
    response_cache_ttl_seconds: int = 3600
    tool_cache_ttl_seconds: int = 3600
    phase1_model: str = "o3-mini"
    extraction_model: str = "gpt-4o-mini"
    coordinator_model: str = "gpt-4o-mini"
    finance_model: str = "gpt-4o-mini"
    hr_model: str = "gpt-4o-mini"
    planner_model: str = "o3-mini"
    executor_model: str = "gpt-4o-mini"
    summarization_model: str = "gpt-4o-mini"
    compaction_char_threshold: int = 5000
    max_recent_messages: int = 8

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "AppSettings":
        root = Path(project_root or os.getcwd()).resolve()
        data_dir = root / "data"
        return cls(
            project_root=root,
            data_dir=data_dir,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
            default_max_turns=int(os.getenv("DEFAULT_MAX_TURNS", "10")),
            circuit_breaker_threshold=int(
                os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3")
            ),
            request_retry_budget=int(os.getenv("REQUEST_RETRY_BUDGET", "5")),
            retry_backoff_base_seconds=float(
                os.getenv("RETRY_BACKOFF_BASE_SECONDS", "0.25")
            ),
            retry_backoff_max_seconds=float(
                os.getenv("RETRY_BACKOFF_MAX_SECONDS", "2.0")
            ),
            context_ttl_steps=int(os.getenv("CONTEXT_TTL_STEPS", "2")),
            max_context_estimated_tokens=int(
                os.getenv("MAX_CONTEXT_ESTIMATED_TOKENS", "700")
            ),
            response_cache_ttl_seconds=int(
                os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600")
            ),
            tool_cache_ttl_seconds=int(
                os.getenv("TOOL_CACHE_TTL_SECONDS", "3600")
            ),
            phase1_model=os.getenv("PHASE1_MODEL", "o3-mini"),
            extraction_model=os.getenv("EXTRACTION_MODEL", "gpt-4o-mini"),
            coordinator_model=os.getenv("COORDINATOR_MODEL", "gpt-4o-mini"),
            finance_model=os.getenv("FINANCE_MODEL", "gpt-4o-mini"),
            hr_model=os.getenv("HR_MODEL", "gpt-4o-mini"),
            planner_model=os.getenv("PLANNER_MODEL", "o3-mini"),
            executor_model=os.getenv("EXECUTOR_MODEL", "gpt-4o-mini"),
            summarization_model=os.getenv("SUMMARIZATION_MODEL", "gpt-4o-mini"),
            compaction_char_threshold=int(
                os.getenv("COMPACTION_CHAR_THRESHOLD", "5000")
            ),
            max_recent_messages=int(os.getenv("MAX_RECENT_MESSAGES", "8")),
        )

    def provider_name(self) -> str:
        if self.openai_api_key:
            return "openai"
        if self.openrouter_api_key:
            return "openrouter"
        return "unconfigured"
