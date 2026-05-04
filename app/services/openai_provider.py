from __future__ import annotations

import logging
from typing import Any

import httpx
import orjson

from app.config import Settings
from app.core.models import UsageRecord
from app.services.concurrency import ConcurrencyQueue, QueueOverloadedError

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._openrouter_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._openai_queue = ConcurrencyQueue(settings.openai_concurrency, settings.max_queue_backlog)

    async def close(self) -> None:
        await self._client.aclose()
        await self._openrouter_client.aclose()

    async def transcribe_audio(
        self,
        wav_bytes: bytes,
        *,
        prompt: str | None = None,
        language: str | None = None,
    ) -> tuple[str, UsageRecord | None]:
        self._require_openai_key("transcription")
        files = {
            "file": ("turn.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "model": self.settings.stt_model,
            "response_format": "json",
            "language": language or self.settings.transcription_language,
            "prompt": prompt or self.settings.transcription_prompt,
        }
        response = await self._post_openai(
            "audio_transcription",
            f"{self.settings.openai_base_url}/audio/transcriptions",
            headers=self._openai_headers(),
            files=files,
            data=data,
        )
        self._raise_for_status(response, "transcription")
        payload = response.json()
        usage = payload.get("usage") or {}
        record = UsageRecord(
            provider="openai",
            model=self.settings.stt_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata=usage,
        )
        return payload.get("text", "").strip(), record

    async def synthesize_speech(self, text: str) -> bytes:
        self._require_openai_key("speech synthesis")
        payload = {
            "model": self.settings.tts_model,
            "voice": self.settings.tts_voice,
            "input": text,
            "instructions": self.settings.tts_style_prompt,
            "response_format": "mp3",
        }
        response = await self._post_openai(
            "audio_speech",
            f"{self.settings.openai_base_url}/audio/speech",
            headers=self._openai_headers(json_body=True),
            json=payload,
        )
        self._raise_for_status(response, "speech synthesis")
        return response.content

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.1,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], UsageRecord | None]:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            "metadata": metadata or {},
        }
        if not self.settings.openai_api_key:
            if not self.settings.openrouter_api_key:
                raise ProviderError("OPENAI_API_KEY or OPENROUTER_API_KEY is required for text generation.")
            return await self._generate_json_openrouter(system_prompt, user_prompt, schema_name, model)
        try:
            response = await self._post_openai(
                "responses_json",
                f"{self.settings.openai_base_url}/responses",
                headers=self._openai_headers(json_body=True),
                json=payload,
            )
            self._raise_for_status(response, "response generation")
            parsed = self._parse_response_json(response.json(), model)
            return parsed
        except ProviderError:
            if not self.settings.openrouter_api_key:
                raise
            logger.warning("openai_json_failed_falling_back_to_openrouter")
            return await self._generate_json_openrouter(system_prompt, user_prompt, schema_name, model)

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.2,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, UsageRecord | None]:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "store": False,
            "metadata": metadata or {},
        }
        if not self.settings.openai_api_key:
            if not self.settings.openrouter_api_key:
                raise ProviderError("OPENAI_API_KEY or OPENROUTER_API_KEY is required for text generation.")
            return await self._generate_text_openrouter(system_prompt, user_prompt, model)
        try:
            response = await self._post_openai(
                "responses_text",
                f"{self.settings.openai_base_url}/responses",
                headers=self._openai_headers(json_body=True),
                json=payload,
            )
            self._raise_for_status(response, "response generation")
            data = response.json()
            return self._extract_text(data), self._usage_from_response(data, model)
        except ProviderError:
            if not self.settings.openrouter_api_key:
                raise
            return await self._generate_text_openrouter(system_prompt, user_prompt, model)

    async def _generate_json_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        model: str,
    ) -> tuple[dict[str, Any], UsageRecord | None]:
        payload = {
            "model": f"openai/{model}",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Return valid JSON for schema {schema_name}.\n" + user_prompt
                    ),
                },
            ],
            "temperature": 0.1,
        }
        response = await self._post_openrouter(
            "openrouter_json",
            f"{self.settings.openrouter_base_url}/chat/completions",
            headers=self._openrouter_headers(),
            json=payload,
        )
        self._raise_for_status(response, "openrouter response generation")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return orjson.loads(content), UsageRecord(
            provider="openrouter",
            model=f"openai/{model}",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata=usage,
        )

    async def _generate_text_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> tuple[str, UsageRecord | None]:
        payload = {
            "model": f"openai/{model}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = await self._post_openrouter(
            "openrouter_text",
            f"{self.settings.openrouter_base_url}/chat/completions",
            headers=self._openrouter_headers(),
            json=payload,
        )
        self._raise_for_status(response, "openrouter response generation")
        data = response.json()
        usage = data.get("usage") or {}
        return data["choices"][0]["message"]["content"].strip(), UsageRecord(
            provider="openrouter",
            model=f"openai/{model}",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata=usage,
        )

    def _parse_response_json(
        self,
        data: dict[str, Any],
        model: str,
    ) -> tuple[dict[str, Any], UsageRecord | None]:
        text = self._extract_text(data)
        try:
            return orjson.loads(text), self._usage_from_response(data, model)
        except orjson.JSONDecodeError as exc:
            raise ProviderError(f"Model returned invalid JSON: {text}") from exc

    def _extract_text(self, data: dict[str, Any]) -> str:
        parts: list[str] = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
        return "".join(parts).strip()

    def _usage_from_response(self, data: dict[str, Any], model: str) -> UsageRecord | None:
        usage = data.get("usage")
        if not usage:
            return None
        return UsageRecord(
            provider="openai",
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata=usage,
        )

    def _openai_headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _openrouter_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    async def _post_openai(self, operation_name: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response, _ = await self._openai_queue.run(
                operation_name,
                lambda: self._client.post(url, **kwargs),
            )
            return response
        except QueueOverloadedError as exc:
            raise ProviderError(str(exc)) from exc

    async def _post_openrouter(self, operation_name: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response, _ = await self._openai_queue.run(
                operation_name,
                lambda: self._openrouter_client.post(url, **kwargs),
            )
            return response
        except QueueOverloadedError as exc:
            raise ProviderError(str(exc)) from exc

    def _require_openai_key(self, capability: str) -> None:
        if not self.settings.openai_api_key:
            raise ProviderError(f"OPENAI_API_KEY is required for {capability}.")

    def _raise_for_status(self, response: httpx.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise ProviderError(f"{operation} failed: {exc.response.status_code} {detail}") from exc
