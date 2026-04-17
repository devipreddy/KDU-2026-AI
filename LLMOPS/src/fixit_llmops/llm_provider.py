from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI

from .models import LLMGeneration, LLMProviderConfig, ModelSpec, TokenUsage


class LLMProvider(Protocol):
    def generate(self, model_alias: str, model: ModelSpec, system_prompt: str, user_prompt: str) -> LLMGeneration:
        ...


class OpenAICompatibleProvider:
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=os.getenv(config.api_key_env, "missing-api-key"),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            default_headers=config.default_headers,
        )

    def generate(self, model_alias: str, model: ModelSpec, system_prompt: str, user_prompt: str) -> LLMGeneration:
        if self.config.dry_run:
            synthetic = (
                f"[dry-run::{model_alias}] Category-aware response generated locally. "
                f"User asked: {user_prompt.strip()[:120]}"
            )
            usage = TokenUsage(prompt_tokens=max(len(system_prompt.split()) + len(user_prompt.split()), 1) * 2, completion_tokens=80)
            return LLMGeneration(text=synthetic, model_alias=model_alias, model_id=model.model_id, usage=usage, raw_response_id=None)

        response = self.client.chat.completions.create(
            model=model.model_id,
            temperature=model.temperature,
            max_tokens=model.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        usage = TokenUsage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        return LLMGeneration(
            text=content.strip(),
            model_alias=model_alias,
            model_id=model.model_id,
            usage=usage,
            raw_response_id=getattr(response, "id", None),
        )


class MockLLMProvider:
    def generate(self, model_alias: str, model: ModelSpec, system_prompt: str, user_prompt: str) -> LLMGeneration:
        del system_prompt
        message = f"{model_alias.upper()} handled: {user_prompt}"
        return LLMGeneration(
            text=message,
            model_alias=model_alias,
            model_id=model.model_id,
            usage=TokenUsage(prompt_tokens=120, completion_tokens=60),
            raw_response_id="mock-response",
        )

