from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Multi-Function AI Assistant"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    allowed_origins: str = "http://localhost:8000"

    llm_provider: str = "openai"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_http_referer: str = "http://localhost:8000"
    openrouter_app_title: str = "Multi-Function AI Assistant"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    llm_cheap_model: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias=AliasChoices("LLM_CHEAP_MODEL", "OPENROUTER_MODEL"),
    )
    llm_strong_model: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias=AliasChoices("LLM_STRONG_MODEL", "OPENROUTER_MODEL"),
    )
    max_tool_iterations: int = 4
    max_memory_messages: int = 8
    cache_ttl_seconds: int = 300
    request_timeout_seconds: int = 30
    circuit_breaker_window_size: int = 6
    circuit_breaker_failure_rate_threshold: float = 0.5
    circuit_breaker_minimum_calls: int = 4
    circuit_breaker_recovery_timeout_seconds: int = 45

    serper_api_key: str | None = None
    serper_base_url: str = "https://google.serper.dev"

    weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    geocoding_base_url: str = "https://geocoding-api.open-meteo.com/v1/search"

    model_input_cost_per_million: float = 0.15
    model_output_cost_per_million: float = 0.60
    model_currency: str = "USD"

    def has_provider(self, provider: Literal["openai", "openrouter"]) -> bool:
        if provider == "openai":
            return bool(self.openai_api_key)
        return bool(self.openrouter_api_key)

    @property
    def ordered_llm_providers(self) -> list[Literal["openai", "openrouter"]]:
        preferred = self.llm_provider.strip().lower()
        if preferred not in {"openai", "openrouter"}:
            preferred = "openai"

        alternate = "openrouter" if preferred == "openai" else "openai"
        providers: list[Literal["openai", "openrouter"]] = []
        for provider in (preferred, alternate):
            if self.has_provider(provider):
                providers.append(provider)
        return providers

    @property
    def primary_provider(self) -> Literal["openai", "openrouter"] | None:
        providers = self.ordered_llm_providers
        return providers[0] if providers else None

    def provider_api_key(self, provider: Literal["openai", "openrouter"]) -> str | None:
        return self.openai_api_key if provider == "openai" else self.openrouter_api_key

    def provider_base_url(self, provider: Literal["openai", "openrouter"]) -> str:
        return self.openai_base_url if provider == "openai" else self.openrouter_base_url

    def provider_headers(
        self,
        provider: Literal["openai", "openrouter"],
    ) -> dict[str, str] | None:
        if provider != "openrouter":
            return None
        return {
            "HTTP-Referer": self.openrouter_http_referer,
            "X-Title": self.openrouter_app_title,
        }

    def resolve_model_for_provider(
        self,
        model: str | None,
        provider: Literal["openai", "openrouter"],
    ) -> str:
        candidate = (model or "").strip()
        if not candidate:
            candidate = self.openai_model if provider == "openai" else self.openrouter_model
        if provider == "openai" and "/" in candidate:
            candidate = candidate.split("/")[-1]
        return candidate

    def provider_default_model(self, provider: Literal["openai", "openrouter"]) -> str:
        raw_model = self.openai_model if provider == "openai" else self.openrouter_model
        return self.resolve_model_for_provider(raw_model, provider)

    @property
    def llm_api_key(self) -> str:
        provider = self.primary_provider
        key = self.provider_api_key(provider) if provider else None
        if not key:
            raise ValueError(
                "No LLM API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )
        return key

    @property
    def llm_base_url(self) -> str:
        provider = self.primary_provider
        if not provider:
            return self.openai_base_url
        return self.provider_base_url(provider)

    @property
    def default_model(self) -> str:
        provider = self.primary_provider
        if not provider:
            return self.resolve_model_for_provider(self.openai_model, "openai")
        return self.provider_default_model(provider)

    @property
    def is_openrouter(self) -> bool:
        return self.primary_provider == "openrouter"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
