from __future__ import annotations

import base64
import logging
from time import perf_counter
from typing import Any, Iterable

from openai import OpenAI

from app.core.config import Settings
from app.services.costs import calculate_cost
from app.services.types import ApiUsageEntry, VisionExtractionResult, VisualDescriptionResult
from app.utils.json_utils import extract_json


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self._openai_client = self._build_openai_client()
        self._openrouter_client = self._build_openrouter_client()

    def extract_image_content(
        self,
        image_bytes: bytes,
        mime_type: str,
        context_label: str,
        operation: str,
    ) -> VisionExtractionResult:
        system_prompt = (
            "You extract accessible content from images and document pages. "
            "Return strict JSON with keys: extracted_text, visual_description, layout_notes, detected_language. "
            "Rules: preserve visible text in natural reading order, describe diagrams/tables/charts for screen readers, "
            "do not invent values, and keep visual_description concise but informative."
        )
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Analyze this {context_label}. Extract readable text and provide an accessibility-focused "
                    "description of important non-text visuals."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                    "detail": self.settings.vision_detail,
                },
            },
        ]
        payload, usage = self._chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            operation=operation,
            mode="vision",
            max_tokens=1200,
        )
        return VisionExtractionResult(
            extracted_text=str(payload.get("extracted_text", "")).strip(),
            visual_description=str(payload.get("visual_description", "")).strip(),
            layout_notes=str(payload.get("layout_notes", "")).strip(),
            detected_language=str(payload.get("detected_language", "unknown")).strip(),
            usage_entry=usage,
        )

    def enrich_document(self, text: str, operation: str, chunk_context: str | None = None) -> tuple[dict[str, Any], ApiUsageEntry]:
        system_prompt = (
            "You produce structured accessibility outputs for extracted content. "
            "Return strict JSON with keys: title, summary, key_points, topic_tags. "
            "summary must be around 150 words unless the user content is extremely short. "
            "key_points must contain 5 to 7 bullets for full documents, or up to 3 bullets for chunk-level summaries. "
            "topic_tags must be short, concrete tags."
        )
        prompt_prefix = f"{chunk_context}\n\n" if chunk_context else ""
        user_prompt = (
            f"{prompt_prefix}Document text:\n{text}\n\n"
            "Return JSON only. Make the summary concise, extract the most decision-relevant points, "
            "and infer sensible topic tags."
        )
        return self._chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            operation=operation,
            mode="text",
            max_tokens=1200,
        )

    def describe_visuals(
        self,
        image_bytes: bytes,
        mime_type: str,
        context_label: str,
        operation: str,
        extracted_text_hint: str | None = None,
    ) -> VisualDescriptionResult:
        system_prompt = (
            "You are generating accessibility descriptions for document visuals. "
            "Return strict JSON with keys: visual_description, layout_notes, contains_meaningful_visuals. "
            "Describe only meaningful non-text visuals such as charts, tables, diagrams, signatures, annotations, or figures. "
            "Ignore decorative logos, borders, and repeated letterheads unless they carry meaning."
        )
        hint = f"\nExisting extracted text:\n{extracted_text_hint}" if extracted_text_hint else ""
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Inspect this {context_label} and describe important non-text visuals for screen-reader users."
                    f"{hint}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}",
                    "detail": self.settings.vision_detail,
                },
            },
        ]
        payload, usage = self._chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            operation=operation,
            mode="vision",
            max_tokens=700,
        )
        meaningful = payload.get("contains_meaningful_visuals", False)
        if isinstance(meaningful, str):
            meaningful = meaningful.strip().lower() in {"true", "yes", "1"}
        return VisualDescriptionResult(
            visual_description=str(payload.get("visual_description", "")).strip(),
            layout_notes=str(payload.get("layout_notes", "")).strip(),
            contains_meaningful_visuals=bool(meaningful),
            usage_entry=usage,
        )

    def create_embeddings(self, texts: list[str], operation: str) -> tuple[list[list[float]], ApiUsageEntry]:
        last_error: Exception | None = None
        for provider_name, client, model_name in self._iter_embedding_clients():
            start = perf_counter()
            try:
                response = client.embeddings.create(model=model_name, input=texts)
                elapsed_ms = int((perf_counter() - start) * 1000)
                usage = self._build_usage_entry(
                    response=response,
                    provider=provider_name,
                    model=model_name,
                    operation=operation,
                    response_ms=elapsed_ms,
                )
                vectors = [item.embedding for item in response.data]
                return vectors, usage
            except Exception as exc:  # pragma: no cover - network/external API branch
                last_error = exc
                self.logger.warning("Embedding request failed via %s: %s", provider_name, exc)
        raise RuntimeError("Embedding generation failed.") from last_error

    def _chat_json(
        self,
        *,
        messages: list[dict[str, Any]],
        operation: str,
        mode: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], ApiUsageEntry]:
        last_error: Exception | None = None
        for provider_name, client, model_name in self._iter_chat_clients(mode):
            start = perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                elapsed_ms = int((perf_counter() - start) * 1000)
                content = response.choices[0].message.content or "{}"
                payload = extract_json(content)
                usage = self._build_usage_entry(
                    response=response,
                    provider=provider_name,
                    model=model_name,
                    operation=operation,
                    response_ms=elapsed_ms,
                )
                return payload, usage
            except Exception as exc:  # pragma: no cover - network/external API branch
                last_error = exc
                self.logger.warning("%s request failed via %s: %s", mode, provider_name, exc)
        raise RuntimeError(f"{mode.capitalize()} request failed.") from last_error

    def _iter_chat_clients(self, mode: str) -> Iterable[tuple[str, OpenAI, str]]:
        if self._openai_client is not None:
            model = self.settings.openai_vision_model if mode == "vision" else self.settings.openai_generation_model
            yield "openai", self._openai_client, model
        if self._openrouter_client is not None:
            model = self.settings.openrouter_vision_model if mode == "vision" else self.settings.openrouter_generation_model
            yield "openrouter", self._openrouter_client, model
        if self._openai_client is None and self._openrouter_client is None:
            raise RuntimeError("No generation provider configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    def _iter_embedding_clients(self) -> Iterable[tuple[str, OpenAI, str]]:
        if self._openai_client is not None:
            yield "openai", self._openai_client, self.settings.openai_embedding_model
        if self._openrouter_client is not None and self.settings.openrouter_embedding_model:
            yield "openrouter", self._openrouter_client, self.settings.openrouter_embedding_model
        if self._openai_client is None and self._openrouter_client is None:
            raise RuntimeError("No embedding provider configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

    def _build_openai_client(self) -> OpenAI | None:
        if not self.settings.openai_api_key:
            return None
        return OpenAI(api_key=self.settings.openai_api_key, max_retries=0, timeout=120.0)

    def _build_openrouter_client(self) -> OpenAI | None:
        if not self.settings.openrouter_api_key:
            return None
        headers = {}
        if self.settings.openrouter_referer:
            headers["HTTP-Referer"] = self.settings.openrouter_referer
        if self.settings.openrouter_app_title:
            headers["X-Title"] = self.settings.openrouter_app_title
        return OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers=headers or None,
            max_retries=0,
            timeout=120.0,
        )

    def _build_usage_entry(
        self,
        *,
        response: Any,
        provider: str,
        model: str,
        operation: str,
        response_ms: int,
    ) -> ApiUsageEntry:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
        estimated_cost = calculate_cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        return ApiUsageEntry(
            operation=operation,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            response_ms=response_ms,
        )
