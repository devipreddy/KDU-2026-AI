from __future__ import annotations

from typing import Any

import tiktoken
from openai import AsyncOpenAI

from app.config import Settings


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.is_configured = bool(settings.ordered_llm_providers)
        self.clients: dict[str, AsyncOpenAI] = {}
        for provider in settings.ordered_llm_providers:
            api_key = settings.provider_api_key(provider)
            if not api_key:
                continue
            self.clients[provider] = AsyncOpenAI(
                api_key=api_key,
                base_url=settings.provider_base_url(provider),
                default_headers=settings.provider_headers(provider),
                timeout=settings.request_timeout_seconds,
            )

    async def create_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> Any:
        if not self.is_configured:
            raise RuntimeError(
                "No LLM API key configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )

        errors: list[str] = []
        last_error: Exception | None = None
        providers = self.settings.ordered_llm_providers

        for index, provider in enumerate(providers):
            client = self.clients.get(provider)
            if client is None:
                continue

            payload: dict[str, Any] = {
                "model": self.settings.resolve_model_for_provider(model, provider),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
                payload["parallel_tool_calls"] = True
            if stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}

            try:
                return await client.chat.completions.create(**payload)
            except Exception as exc:
                last_error = exc
                errors.append(f"{provider}: {exc}")
                if index < len(providers) - 1:
                    continue

        if last_error is not None:
            raise RuntimeError(" | ".join(errors)) from last_error
        raise RuntimeError("No LLM client is available for the configured providers.")

    @staticmethod
    def usage_to_dict(usage: Any | None) -> dict[str, int]:
        if not usage:
            return {"input": 0, "output": 0, "total": 0}
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
        return {
            "input": int(input_tokens),
            "output": int(output_tokens),
            "total": int(total_tokens),
        }

    @staticmethod
    def merge_usage(current: dict[str, int], incoming: dict[str, int]) -> dict[str, int]:
        return {
            "input": current.get("input", 0) + incoming.get("input", 0),
            "output": current.get("output", 0) + incoming.get("output", 0),
            "total": current.get("total", 0) + incoming.get("total", 0),
        }

    def estimate_tokens(self, text: str, model: str) -> int:
        if not text:
            return 0
        encoding_name = "o200k_base" if "4o" in model else "cl100k_base"
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
