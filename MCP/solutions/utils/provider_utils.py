"""
Provider utilities for OpenAI- and OpenRouter-compatible model setup.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENAI_PROVIDER = "openai"
OPENROUTER_PROVIDER = "openrouter"

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    chat_model: str
    embedding_model: str
    base_url: Optional[str] = None
    default_headers: Optional[Dict[str, str]] = None


def _pick_provider(explicit_provider=None):
    provider = explicit_provider or os.environ.get("LLM_PROVIDER") or os.environ.get("MODEL_PROVIDER")
    if provider:
        provider = provider.strip().lower()
        if provider in {OPENAI_PROVIDER, OPENROUTER_PROVIDER}:
            return provider

    if os.environ.get("OPENROUTER_API_KEY"):
        return OPENROUTER_PROVIDER

    return OPENAI_PROVIDER


def build_provider_config(explicit_api_key=None, explicit_provider=None):
    """Build provider configuration from the current environment."""
    provider = _pick_provider(explicit_provider)

    if provider == OPENROUTER_PROVIDER:
        api_key = explicit_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        site_url = os.environ.get("OPENROUTER_SITE_URL") or os.environ.get("OPENROUTER_HTTP_REFERER")
        app_name = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPENROUTER_TITLE")
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-OpenRouter-Title"] = app_name

        return ProviderConfig(
            provider=provider,
            api_key=api_key,
            chat_model=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_CHAT_MODEL),
            embedding_model=os.environ.get(
                "OPENROUTER_EMBEDDING_MODEL",
                DEFAULT_OPENROUTER_EMBEDDING_MODEL,
            ),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            default_headers=default_headers or None,
        )

    api_key = explicit_api_key or os.environ.get("OPENAI_API_KEY", "")
    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        chat_model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        embedding_model=os.environ.get(
            "OPENAI_EMBEDDING_MODEL",
            DEFAULT_OPENAI_EMBEDDING_MODEL,
        ),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        default_headers=None,
    )


def init_chat_model(config: ProviderConfig, temperature=0):
    """Initialize a chat model for the configured provider."""
    if not config.api_key:
        return None

    return ChatOpenAI(
        model=config.chat_model,
        temperature=temperature,
        openai_api_key=config.api_key,
        openai_api_base=config.base_url,
        default_headers=config.default_headers,
    )


def init_embedding_model(config: ProviderConfig):
    """Initialize an embedding model for the configured provider."""
    if not config.api_key:
        return None

    return OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=config.api_key,
        openai_api_base=config.base_url,
        default_headers=config.default_headers,
    )


def init_langchain_provider_components(explicit_api_key=None, explicit_provider=None, temperature=0):
    """Return ``(embeddings, llm, config)`` for the active provider."""
    config = build_provider_config(explicit_api_key, explicit_provider)
    embeddings = init_embedding_model(config)
    llm = init_chat_model(config, temperature=temperature)
    return embeddings, llm, config
