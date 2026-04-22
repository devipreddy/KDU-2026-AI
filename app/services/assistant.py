from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from app.cache import TTLCacheStore
from app.config import Settings
from app.costs import build_cost_breakdown
from app.memory import SessionMemoryStore
from app.metrics import MetricsStore
from app.resilience import AsyncCircuitBreaker
from app.schemas import AssistantState, ChatRequest, ChatResponse, TokenUsage, build_initial_state
from app.services.graph import AssistantGraph
from app.services.llm import LLMService
from app.services.prompts import SUMMARY_SYSTEM_PROMPT
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.search import SearchTool
from app.tools.weather import WeatherTool


class AssistantService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_store = TTLCacheStore(ttl_seconds=settings.cache_ttl_seconds)
        self.memory_store = SessionMemoryStore(max_messages=settings.max_memory_messages)
        self.metrics_store = MetricsStore()
        self.http_client = httpx.AsyncClient(
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": settings.app_name},
        )
        self.circuit_breakers = {
            "search": AsyncCircuitBreaker(
                name="serper",
                window_size=settings.circuit_breaker_window_size,
                failure_rate_threshold=settings.circuit_breaker_failure_rate_threshold,
                minimum_calls=settings.circuit_breaker_minimum_calls,
                recovery_timeout_seconds=settings.circuit_breaker_recovery_timeout_seconds,
            ),
            "weather": AsyncCircuitBreaker(
                name="weather",
                window_size=settings.circuit_breaker_window_size,
                failure_rate_threshold=settings.circuit_breaker_failure_rate_threshold,
                minimum_calls=settings.circuit_breaker_minimum_calls,
                recovery_timeout_seconds=settings.circuit_breaker_recovery_timeout_seconds,
            ),
        }
        self.llm_service = LLMService(settings)
        self.tool_registry = ToolRegistry(
            [
                CalculatorTool(),
                WeatherTool(settings, self.http_client, self.circuit_breakers["weather"]),
                SearchTool(settings, self.http_client, self.circuit_breakers["search"]),
            ]
        )
        self.assistant_graph = AssistantGraph(
            settings=settings,
            cache_store=self.cache_store,
            memory_store=self.memory_store,
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
        )

    async def close(self) -> None:
        await self.http_client.aclose()

    async def chat(self, payload: ChatRequest) -> ChatResponse:
        state = build_initial_state(payload)
        final_state = await self.assistant_graph.graph.ainvoke(state)
        return await self._finalize_response(final_state)

    async def stream_chat(self, payload: ChatRequest) -> AsyncIterator[str]:
        initial_state = build_initial_state(payload)
        merged_state: AssistantState = dict(initial_state)
        response_text = ""

        try:
            yield self._sse(
                "start",
                {
                    "request_id": initial_state["request_id"],
                    "session_id": initial_state["session_id"],
                },
            )

            async for chunk in self.assistant_graph.graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                for node_name, updates in chunk.items():
                    updates = updates or {}
                    merged_state.update(updates)
                    if updates.get("error"):
                        yield self._sse("error", {"message": updates["error"]})
                        return
                    if updates.get("planner_error"):
                        yield self._sse(
                            "warning",
                            {
                                "stage": "planner",
                                "message": updates["planner_error"],
                                "fallback_mode": updates.get("fallback_mode"),
                            },
                        )

                    if node_name == "route":
                        yield self._sse(
                            "route",
                            {
                                "route": updates.get("route"),
                                "intent": updates.get("intent"),
                                "model": updates.get("llm_model"),
                                "complexity": updates.get("complexity"),
                                "tool": updates.get("tool_name"),
                            },
                        )
                        if updates.get("tool_name"):
                            yield self._sse(
                                "tool_call",
                                {
                                    "tool": updates["tool_name"],
                                    "input": updates.get("tool_input"),
                                },
                            )

                    if node_name == "cache_lookup" and updates.get("cache_hit"):
                        yield self._sse(
                            "cache_hit",
                            {
                                "tool": merged_state.get("tool_name"),
                                "cache_key": updates.get("cache_key"),
                            },
                        )

                    if node_name in {"execute_direct_tool", "llm_tool_loop"} and updates.get(
                        "tool_names"
                    ):
                        yield self._sse(
                            "tool_result",
                            {
                                "tools": self._unique_tool_names(updates.get("tool_names")),
                                "output": updates.get("tool_output"),
                            },
                        )

            if merged_state.get("normalized_tool_output") and not merged_state.get("llm_response"):
                yield self._sse(
                    "tool_result",
                    {
                        "tools": self._unique_tool_names(merged_state.get("tool_names")),
                        "output": merged_state.get("normalized_tool_output"),
                    },
                )
                try:
                    summary_messages = self._build_summary_messages(merged_state)
                    stream = await self.llm_service.create_completion(
                        model=merged_state.get("llm_model") or self.settings.default_model,
                        messages=summary_messages,
                        stream=True,
                        temperature=0.2,
                        max_tokens=700,
                    )
                    extra_usage = {"input": 0, "output": 0, "total": 0}
                    async for part in stream:
                        if getattr(part, "usage", None):
                            extra_usage = self.llm_service.usage_to_dict(part.usage)
                        if not getattr(part, "choices", None):
                            continue
                        delta = part.choices[0].delta
                        token = getattr(delta, "content", None)
                        if token:
                            response_text += token
                            yield self._sse("token", {"text": token})
                    if not any(extra_usage.values()):
                        extra_usage = self._estimate_usage_from_messages(
                            summary_messages,
                            response_text,
                            merged_state.get("llm_model") or self.settings.default_model,
                        )
                    merged_state["tokens_used"] = self.llm_service.merge_usage(
                        merged_state.get("tokens_used", {"input": 0, "output": 0, "total": 0}),
                        extra_usage,
                    )
                    merged_state["llm_calls"] = merged_state.get("llm_calls", 0) + 1
                except Exception as exc:
                    merged_state["summary_error"] = str(exc)
                    merged_state["fallback_mode"] = "tool_only"
                    yield self._sse(
                        "warning",
                        {
                            "stage": "summary",
                            "message": str(exc),
                            "fallback_mode": "tool_only",
                        },
                    )
                    response_text = self._render_fallback_response(merged_state)
                    for chunk_text in self._chunk_text(response_text):
                        yield self._sse("token", {"text": chunk_text})
            else:
                response_text = merged_state.get("llm_response") or ""
                for chunk_text in self._chunk_text(response_text):
                    yield self._sse("token", {"text": chunk_text})

            chat_response = await self._finalize_response(
                merged_state,
                response_override=response_text,
            )
            yield self._sse("usage", chat_response.tokens.model_dump())
            yield self._sse("cost", chat_response.cost.model_dump())
            yield self._sse(
                "end",
                {
                    "status": "complete",
                    "request_id": chat_response.request_id,
                    "route": chat_response.route,
                    "tools": self._unique_tool_names(chat_response.tool_used),
                    "latency_ms": chat_response.latency_ms,
                    "fallback_mode": merged_state.get("fallback_mode"),
                },
            )
        except Exception as exc:
            yield self._sse("error", {"message": str(exc)})

    async def _finalize_response(
        self,
        state: AssistantState,
        *,
        response_override: str | None = None,
    ) -> ChatResponse:
        final_state: AssistantState = dict(state)
        final_response = (response_override or final_state.get("llm_response") or "").strip()

        if final_state.get("error"):
            raise RuntimeError(final_state["error"])

        if final_state.get("normalized_tool_output") and not final_response:
            try:
                summary_messages = self._build_summary_messages(final_state)
                completion = await self.llm_service.create_completion(
                    model=final_state.get("llm_model") or self.settings.default_model,
                    messages=summary_messages,
                    temperature=0.2,
                    max_tokens=700,
                )
                final_response = (completion.choices[0].message.content or "").strip()
                usage = self.llm_service.usage_to_dict(completion.usage)
                final_state["tokens_used"] = self.llm_service.merge_usage(
                    final_state.get("tokens_used", {"input": 0, "output": 0, "total": 0}),
                    usage,
                )
                final_state["llm_calls"] = final_state.get("llm_calls", 0) + 1
            except Exception as exc:
                final_state["summary_error"] = str(exc)
                final_state["fallback_mode"] = "tool_only"
                final_response = self._render_fallback_response(final_state)

        if not final_response:
            raise RuntimeError("Assistant produced an empty response.")

        completed_at = datetime.now(timezone.utc)
        latency_ms = int(
            (completed_at - final_state["started_at"]).total_seconds() * 1000
        )
        final_state["completed_at"] = completed_at
        final_state["latency_ms"] = latency_ms

        tokens = TokenUsage(**final_state.get("tokens_used", {}))
        cost = build_cost_breakdown(tokens, self.settings)

        self.memory_store.append_turn(
            final_state["session_id"],
            final_state["user_query"],
            final_response,
        )
        self.metrics_store.record_request(
            latency_ms=latency_ms,
            input_tokens=tokens.input,
            output_tokens=tokens.output,
            cache_hit=bool(final_state.get("cache_hit")),
            llm_calls=final_state.get("llm_calls", 0),
            tool_names=self._unique_tool_names(final_state.get("tool_names")),
        )

        return ChatResponse(
            request_id=final_state["request_id"],
            response=final_response,
            route=final_state.get("route", "llm"),
            tool_used=self._unique_tool_names(final_state.get("tool_names")),
            model_used=final_state.get("llm_model") or self.settings.default_model,
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms,
            cache_hit=bool(final_state.get("cache_hit")),
            timestamp=completed_at.isoformat(),
        )

    def get_metrics(self) -> dict[str, object]:
        return self.metrics_store.snapshot()

    def get_dependency_status(self) -> dict[str, str]:
        llm_status = (
            "configured"
            if self.settings.openrouter_api_key or self.settings.openai_api_key
            else "missing"
        )
        serper_state = self.circuit_breakers["search"].snapshot().state
        weather_state = self.circuit_breakers["weather"].snapshot().state
        serper_prefix = "configured" if self.settings.serper_api_key else "missing"
        return {
            "llm": llm_status,
            "serper": f"{serper_prefix}:{serper_state}",
            "weather": f"ok:{weather_state}",
        }

    def get_circuit_breaker_status(self) -> dict[str, dict[str, object]]:
        return {
            name: snapshot.__dict__
            for name, snapshot in (
                ("search", self.circuit_breakers["search"].snapshot()),
                ("weather", self.circuit_breakers["weather"].snapshot()),
            )
        }

    def invalidate_cache(self, key: str) -> bool:
        return self.cache_store.invalidate(key)

    @staticmethod
    def _unique_tool_names(tool_names: Any) -> list[str]:
        if not tool_names:
            return []
        if isinstance(tool_names, str):
            candidates = [tool_names]
        else:
            try:
                candidates = list(tool_names)
            except TypeError:
                return []
        cleaned = [str(name) for name in candidates if name]
        return list(dict.fromkeys(cleaned))

    def _build_summary_messages(self, state: AssistantState) -> list[dict[str, str]]:
        tool_payload = json.dumps(
            state.get("normalized_tool_output"),
            indent=2,
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User question: {state['cleaned_query']}\n\n"
                    f"Tool outputs:\n{tool_payload}"
                ),
            },
        ]

    def _render_fallback_response(self, state: AssistantState) -> str:
        normalized = state.get("normalized_tool_output")
        if not normalized:
            return "I could not build a response from the available tool output."

        items = normalized if isinstance(normalized, list) else [normalized]
        parts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            parts.append(self._render_tool_item_fallback(item))

        rendered = "\n\n".join(part for part in parts if part.strip())
        return rendered or "I could not build a response from the available tool output."

    @staticmethod
    def _render_tool_item_fallback(item: dict[str, Any]) -> str:
        tool_name = item.get("tool_name")
        status = item.get("status")
        error = item.get("error")
        if status == "error" and error:
            return f"{tool_name or 'Tool'} error: {error}"

        if tool_name == "calculator":
            expression = item.get("expression", "expression")
            exact_result = item.get("exact_result")
            approximate_result = item.get("approximate_result")
            if exact_result and approximate_result and exact_result != approximate_result:
                return (
                    f"Calculation for {expression}: exact result {exact_result}; "
                    f"approximate result {approximate_result}."
                )
            if approximate_result:
                return f"Calculation for {expression}: {approximate_result}."
            return f"Calculation for {expression} completed."

        if tool_name == "weather":
            location = item.get("location") or {}
            current = item.get("current") or {}
            place_bits = [
                str(location.get("name") or "").strip(),
                str(location.get("admin1") or "").strip(),
                str(location.get("country") or "").strip(),
            ]
            place = ", ".join(bit for bit in place_bits if bit)
            temperature = current.get("temperature")
            units = (current.get("units") or {}).get("temperature", "")
            description = current.get("description", "Current conditions available")
            humidity = current.get("relative_humidity")
            wind = current.get("wind_speed")
            details = [f"{description}"]
            if temperature is not None:
                details.append(f"temperature {temperature} {units}".strip())
            if humidity is not None:
                details.append(f"humidity {humidity}%")
            if wind is not None:
                details.append(f"wind {wind} km/h")
            prefix = f"Weather for {place}: " if place else "Weather update: "
            return prefix + ", ".join(details) + "."

        if tool_name == "search":
            query = item.get("query", "your query")
            answer_box = item.get("answer_box") or {}
            lines = [f"Search results for {query}:"]
            if isinstance(answer_box, dict):
                answer = answer_box.get("answer") or answer_box.get("snippet")
                if answer:
                    lines.append(str(answer))
            results = item.get("results") or []
            if isinstance(results, list) and results:
                for index, result in enumerate(results[:3], start=1):
                    if not isinstance(result, dict):
                        continue
                    title = result.get("title") or "Untitled result"
                    link = result.get("link") or ""
                    snippet = result.get("snippet") or ""
                    line = f"{index}. {title}"
                    if link:
                        line += f" - {link}"
                    if snippet:
                        line += f" - {snippet}"
                    lines.append(line)
            news = item.get("news") or []
            if isinstance(news, list) and news:
                lines.append("News:")
                for news_item in news[:3]:
                    if not isinstance(news_item, dict):
                        continue
                    title = news_item.get("title") or "Untitled news item"
                    link = news_item.get("link") or ""
                    line = f"- {title}"
                    if link:
                        line += f" - {link}"
                    lines.append(line)
            return "\n".join(lines)

        return json.dumps(item, ensure_ascii=False, indent=2)

    def _estimate_usage_from_messages(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        model: str,
    ) -> dict[str, int]:
        prompt_tokens = sum(
            self.llm_service.estimate_tokens(str(message.get("content", "")), model)
            for message in messages
        )
        completion_tokens = self.llm_service.estimate_tokens(response_text, model)
        return {
            "input": prompt_tokens,
            "output": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }

    @staticmethod
    def _chunk_text(text: str, size: int = 32) -> list[str]:
        if not text:
            return []
        return [text[index : index + size] for index in range(0, len(text), size)]

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
