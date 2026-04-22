from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.schemas import AssistantState
from app.services.prompts import PLANNER_SYSTEM_PROMPT
from app.services.router import decide_route, detect_tool_domains, extract_weather_location, sanitize_query
from app.tools.base import ToolExecutionError, ToolRegistry


class AssistantGraph:
    def __init__(
        self,
        *,
        settings: Any,
        cache_store: Any,
        memory_store: Any,
        llm_service: Any,
        tool_registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.cache_store = cache_store
        self.memory_store = memory_store
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(AssistantState)
        workflow.add_node("sanitize", self.sanitize_node)
        workflow.add_node("guardrails", self.guardrails_node)
        workflow.add_node("load_memory", self.load_memory_node)
        workflow.add_node("route", self.route_node)
        workflow.add_node("cache_lookup", self.cache_lookup_node)
        workflow.add_node("execute_direct_tool", self.execute_direct_tool_node)
        workflow.add_node("llm_tool_loop", self.llm_tool_loop_node)
        workflow.add_node("normalize", self.normalize_node)

        workflow.add_edge(START, "sanitize")
        workflow.add_edge("sanitize", "guardrails")
        workflow.add_edge("guardrails", "load_memory")
        workflow.add_edge("load_memory", "route")
        workflow.add_conditional_edges(
            "route",
            self.after_route,
            {
                "cache_lookup": "cache_lookup",
                "llm_tool_loop": "llm_tool_loop",
            },
        )
        workflow.add_conditional_edges(
            "cache_lookup",
            self.after_cache,
            {
                "normalize": "normalize",
                "execute_direct_tool": "execute_direct_tool",
            },
        )
        workflow.add_edge("execute_direct_tool", "normalize")
        workflow.add_edge("llm_tool_loop", "normalize")
        workflow.add_edge("normalize", END)
        return workflow.compile()

    async def sanitize_node(self, state: AssistantState) -> dict[str, Any]:
        return {"cleaned_query": sanitize_query(state["user_query"])}

    async def guardrails_node(self, state: AssistantState) -> dict[str, Any]:
        query = state["cleaned_query"]
        if not query:
            return {"error": "Query cannot be empty after sanitization."}
        if len(query) > 4000:
            return {"error": "Query is too long."}
        return {}

    async def load_memory_node(self, state: AssistantState) -> dict[str, Any]:
        memory_context = self.memory_store.get(state["session_id"])
        return {"memory_context": memory_context}

    async def route_node(self, state: AssistantState) -> dict[str, Any]:
        decision = decide_route(state["cleaned_query"])
        detected_domains = sorted(detect_tool_domains(state["cleaned_query"]))
        selected_model = (
            self.settings.llm_strong_model
            if decision.complexity == "high"
            else self.settings.llm_cheap_model
        )
        primary_provider = self.settings.primary_provider or "openai"
        llm_model = self.settings.resolve_model_for_provider(selected_model, primary_provider)
        return {
            "route": decision.route,
            "intent": decision.intent,
            "detected_domains": detected_domains,
            "tool_name": decision.tool_name,
            "tool_input": decision.tool_input,
            "confidence": decision.confidence,
            "complexity": decision.complexity,
            "llm_model": llm_model,
        }

    def after_route(self, state: AssistantState) -> str:
        return "cache_lookup" if state.get("route") == "tool" else "llm_tool_loop"

    async def cache_lookup_node(self, state: AssistantState) -> dict[str, Any]:
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input")
        if not tool_name or not tool_input:
            return {}

        cache_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        cached = self.cache_store.get(cache_key)
        if cached:
            return {
                "tool_output": cached,
                "tool_names": [tool_name],
                "cache_hit": True,
                "cache_key": cache_key,
            }
        return {"cache_key": cache_key}

    def after_cache(self, state: AssistantState) -> str:
        return "normalize" if state.get("cache_hit") else "execute_direct_tool"

    async def execute_direct_tool_node(self, state: AssistantState) -> dict[str, Any]:
        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input") or {}
        if not tool_name:
            return {"error": "No deterministic tool selected."}

        try:
            result = await self.tool_registry.execute(tool_name, tool_input)
        except ToolExecutionError as exc:
            result = {"tool_name": tool_name, "status": "error", "error": str(exc)}
        if state.get("cache_key") and result.get("status") == "success":
            self.cache_store.set(state["cache_key"], result)
        return {"tool_output": result, "tool_names": [tool_name]}

    async def llm_tool_loop_node(self, state: AssistantState) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            *state.get("memory_context", []),
            {"role": "user", "content": state["cleaned_query"]},
        ]

        tokens_used = dict(state.get("tokens_used", {"input": 0, "output": 0, "total": 0}))
        llm_calls = 0
        tool_outputs: list[dict[str, Any]] = []
        tool_names: list[str] = []
        final_response: str | None = None

        for _ in range(self.settings.max_tool_iterations):
            try:
                completion = await self.llm_service.create_completion(
                    model=state["llm_model"],
                    messages=messages,
                    tools=self.tool_registry.schemas(),
                    temperature=0.1,
                    max_tokens=600,
                )
            except Exception as exc:
                fallback = await self._heuristic_multi_tool_fallback(state, str(exc))
                if fallback:
                    return fallback
                return {
                    "error": f"LLM planner failed: {exc}",
                    "tool_output": tool_outputs or None,
                    "tool_names": tool_names,
                    "tokens_used": tokens_used,
                    "llm_calls": llm_calls,
                }
            llm_calls += 1
            usage = self.llm_service.usage_to_dict(completion.usage)
            tokens_used = self.llm_service.merge_usage(tokens_used, usage)

            choice = completion.choices[0].message
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if choice.content:
                assistant_message["content"] = choice.content
            if choice.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in choice.tool_calls
                ]
            messages.append(assistant_message)

            if not choice.tool_calls:
                final_response = (choice.content or "").strip()
                break

            for tool_call in choice.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                try:
                    result = await self.tool_registry.execute(
                        tool_call.function.name,
                        arguments,
                    )
                except ToolExecutionError as exc:
                    result = {
                        "tool_name": tool_call.function.name,
                        "status": "error",
                        "error": str(exc),
                    }

                tool_outputs.append(result)
                tool_names.append(tool_call.function.name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(result),
                    }
                )

        return {
            "planner_messages": messages,
            "messages": messages,
            "tool_output": tool_outputs or None,
            "tool_names": tool_names,
            "llm_response": final_response,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "fallback_mode": None,
            "planner_error": None,
            "route": "hybrid" if tool_outputs else "llm",
        }

    async def _heuristic_multi_tool_fallback(
        self,
        state: AssistantState,
        llm_error: str,
    ) -> dict[str, Any] | None:
        domains = list(state.get("detected_domains") or [])
        query = state["cleaned_query"]
        if len(domains) < 2 and state.get("intent") != "multi_tool":
            return None

        tool_outputs: list[dict[str, Any]] = []
        tool_names: list[str] = []

        if "search" in domains:
            search_args = {"query": query, "max_results": 5}
            try:
                result = await self.tool_registry.execute("search", search_args)
            except ToolExecutionError as exc:
                result = {"tool_name": "search", "status": "error", "error": str(exc)}
            tool_outputs.append(result)
            tool_names.append("search")

        if "weather" in domains:
            extracted = extract_weather_location(query)
            location, units = extracted if extracted else ("Delhi", "celsius")
            try:
                result = await self.tool_registry.execute(
                    "weather",
                    {"location": location, "units": units},
                )
            except ToolExecutionError as exc:
                result = {"tool_name": "weather", "status": "error", "error": str(exc)}
            tool_outputs.append(result)
            tool_names.append("weather")

        if "calculator" in domains and state.get("tool_input"):
            try:
                result = await self.tool_registry.execute(
                    "calculator",
                    state["tool_input"],
                )
            except ToolExecutionError as exc:
                result = {"tool_name": "calculator", "status": "error", "error": str(exc)}
            tool_outputs.append(result)
            tool_names.append("calculator")

        if not tool_outputs:
            return None

        return {
            "tool_output": tool_outputs,
            "tool_names": tool_names,
            "llm_response": None,
            "tokens_used": dict(state.get("tokens_used", {"input": 0, "output": 0, "total": 0})),
            "llm_calls": 0,
            "fallback_mode": "tool_only",
            "route": "hybrid",
            "messages": [],
            "planner_messages": [],
            "planner_error": llm_error,
        }

    async def normalize_node(self, state: AssistantState) -> dict[str, Any]:
        tool_output = state.get("tool_output")
        if not tool_output:
            return {}

        if isinstance(tool_output, list):
            normalized = [self._normalize_tool_item(item) for item in tool_output]
        else:
            normalized = self._normalize_tool_item(tool_output)
        return {"normalized_tool_output": normalized}

    @staticmethod
    def _normalize_tool_item(item: dict[str, Any]) -> dict[str, Any]:
        tool_name = item.get("tool_name")
        if tool_name == "search":
            return {
                "tool_name": "search",
                "status": item.get("status"),
                "query": item.get("query"),
                "answer_box": item.get("answer_box"),
                "results": item.get("results", [])[:3],
                "news": item.get("news", [])[:3],
                "error": item.get("error"),
            }
        if tool_name == "weather":
            return {
                "tool_name": "weather",
                "status": item.get("status"),
                "location": item.get("location"),
                "current": item.get("current"),
                "error": item.get("error"),
            }
        if tool_name == "calculator":
            return {
                "tool_name": "calculator",
                "status": item.get("status"),
                "expression": item.get("expression"),
                "exact_result": item.get("exact_result"),
                "approximate_result": item.get("approximate_result"),
                "is_numeric": item.get("is_numeric"),
                "error": item.get("error"),
            }
        return item
