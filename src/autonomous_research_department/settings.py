from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from autonomous_research_department.runtime import bootstrap_environment, project_root


bootstrap_environment()
load_dotenv()


class ProviderConfig(BaseModel):
    provider: str = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 90
    temperature: float = 0.1


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    serper_api_key: str | None = Field(default=None, alias="SERPER_API_KEY")

    primary_model: str = Field(default="gpt-4o-mini", alias="PRIMARY_MODEL")
    manager_model: str = Field(default="gpt-4o-mini", alias="MANAGER_MODEL")
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )
    model_temperature: float = Field(default=0.1, alias="MODEL_TEMPERATURE")
    request_timeout_seconds: int = Field(
        default=90,
        alias="REQUEST_TIMEOUT_SECONDS",
    )

    intermittent_tool_failure_rate: float = Field(
        default=0.5,
        alias="INTERMITTENT_TOOL_FAILURE_RATE",
    )
    intermittent_tool_seed: int | None = Field(
        default=None,
        alias="INTERMITTENT_TOOL_SEED",
    )
    flow_max_retries: int = Field(default=2, alias="FLOW_MAX_RETRIES")

    crew_verbose: bool = True
    agent_max_iter: int = 6
    hierarchical_planning: bool = True

    @property
    def root_dir(self) -> Path:
        return project_root()

    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def memory_dir(self) -> Path:
        return self.root_dir / ".crewai" / "memory"

    @property
    def localapp_dir(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA", self.root_dir / ".localapp"))

    def primary_llm_config(self) -> ProviderConfig:
        if self.openai_api_key:
            return ProviderConfig(
                provider="openai",
                model=self.primary_model,
                api_key=self.openai_api_key,
                timeout=self.request_timeout_seconds,
                temperature=self.model_temperature,
            )

        if self.openrouter_api_key:
            model = (
                self.openrouter_model
                if "/" in self.openrouter_model
                else f"openai/{self.openrouter_model}"
            )
            return ProviderConfig(
                provider="openai",
                model=model,
                api_key=self.openrouter_api_key,
                base_url=self.openrouter_base_url,
                timeout=self.request_timeout_seconds,
                temperature=self.model_temperature,
            )

        return ProviderConfig(
            provider="openai",
            model=self.primary_model,
            timeout=self.request_timeout_seconds,
            temperature=self.model_temperature,
        )

    def manager_llm_config(self) -> ProviderConfig:
        config = self.primary_llm_config()
        manager_model = self.manager_model
        if config.base_url and "/" not in manager_model:
            manager_model = f"openai/{manager_model}"
        return config.model_copy(update={"model": manager_model})

    def ensure_directories(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.localapp_dir.mkdir(parents=True, exist_ok=True)

    def validate_required_keys(self) -> None:
        missing: list[str] = []
        if not (self.openai_api_key or self.openrouter_api_key):
            missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
        if not self.serper_api_key:
            missing.append("SERPER_API_KEY")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment values: {joined}")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_directories()
    return settings

