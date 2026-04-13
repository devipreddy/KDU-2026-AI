from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _normalize_openrouter_model(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "openai/gpt-4o-mini"
    return cleaned if "/" in cleaned else f"openai/{cleaned}"


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_timeout_seconds: float
    openrouter_max_retries: int
    openrouter_app_url: str
    openrouter_app_title: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool
    market_data_provider: str
    checkpoint_db_path: Path
    artifacts_dir: Path
    default_user_id: str
    default_base_currency: str
    intent_parser_allow_fallback: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        raw_model = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"))
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "your-openrouter-api-key")),
            openrouter_model=_normalize_openrouter_model(raw_model),
            openrouter_timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))),
            openrouter_max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", os.getenv("OPENAI_MAX_RETRIES", "1"))),
            openrouter_app_url=os.getenv("OPENROUTER_APP_URL", "https://localhost"),
            openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", "stocks-langraph"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "stocks-langraph"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
            market_data_provider=os.getenv("MARKET_DATA_PROVIDER", "mock").lower(),
            checkpoint_db_path=Path(os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            default_user_id=os.getenv("DEFAULT_USER_ID", "demo-user"),
            default_base_currency=os.getenv("DEFAULT_BASE_CURRENCY", "INR").upper(),
            intent_parser_allow_fallback=os.getenv("INTENT_PARSER_ALLOW_FALLBACK", "false").lower() == "true",
        )

    def prepare_paths(self) -> None:
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def configure_runtime_env(self) -> None:
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_tracing else "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.langsmith_tracing else "false"
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key) and self.openrouter_api_key != "your-openrouter-api-key"

    @property
    def llm_enabled(self) -> bool:
        return self.openrouter_enabled

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key) and self.langsmith_api_key != "your-langsmith-api-key"


def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.prepare_paths()
    settings.configure_runtime_env()
    return settings
