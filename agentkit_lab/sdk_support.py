"""Helpers for importing and configuring the OpenAI Agents SDK at runtime."""

from __future__ import annotations

import os
from typing import Any, Generic, TypeVar

from .exceptions import MissingDependencyError, ProviderConfigurationError
from .settings import AppSettings

ContextT = TypeVar("ContextT")


class RunContextWrapper(Generic[ContextT]):
    """Fallback generic used for deferred RunContextWrapper annotations."""


def import_agents_sdk() -> Any:
    try:
        import agents
    except ImportError as exc:
        raise MissingDependencyError(
            "The live workflows require `openai-agents`. Install project dependencies first."
        ) from exc
    return agents


def bind_run_context_wrapper(globalns: dict[str, Any]) -> None:
    """Bind the live SDK RunContextWrapper into a module global namespace."""

    globalns["RunContextWrapper"] = import_agents_sdk().RunContextWrapper


def configure_provider(settings: AppSettings) -> str:
    agents = import_agents_sdk()
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise MissingDependencyError(
            "The live workflows require the OpenAI Python client that ships with `openai-agents`."
        ) from exc

    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
        return "openai"

    if settings.openrouter_api_key:
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        agents.set_default_openai_client(client)
        return "openrouter"

    raise ProviderConfigurationError(
        "No live provider is configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
    )
