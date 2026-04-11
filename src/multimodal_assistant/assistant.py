from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import ToolRuntime, tool
from langchain_openrouter import ChatOpenRouter

from multimodal_assistant.config import Settings
from multimodal_assistant.image_analysis import ImageAnalyzer
from multimodal_assistant.schemas import (
    AssistantInvocationResult,
    AssistantResponsePayload,
    AssistantRuntimeContext,
    ImageAnalysisResult,
    WeatherResult,
)
from multimodal_assistant.weather import WeatherClient


BASE_SYSTEM_PROMPT = """
You are a production-grade multimodal assistant.

Requirements:
- Use hidden user context to personalize answers.
- Do not ask for the user's location if one is already available in context.
- Use tools when the request depends on current weather or uploaded image analysis.
- Return a response that cleanly matches the required structured output schema.
- Leave weather fields null when weather is not relevant.
- Leave image fields null or empty when image analysis is not relevant.
""".strip()


class AssistantRunner(Protocol):
    def respond(
        self,
        message: str,
        runtime_context: AssistantRuntimeContext,
    ) -> AssistantInvocationResult: ...

    def close(self) -> None: ...


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return " ".join(chunks).strip()
    return str(content)


def _latest_user_text(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return _content_to_text(message.content)
    return ""


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _mode_instruction(mode: str) -> str:
    if mode == "expert":
        return "Respond in a technical, concise, implementation-focused style."
    if mode == "child":
        return "Respond simply, warmly, and using very easy language."
    return "Respond clearly and professionally."


def _looks_like_weather_request(text: str) -> bool:
    lowered = text.lower()
    keywords = ("weather", "temperature", "forecast", "rain", "snow", "climate", "hot", "cold")
    return any(keyword in lowered for keyword in keywords)


def _looks_like_image_request(text: str) -> bool:
    lowered = text.lower()
    keywords = ("image", "photo", "picture", "describe", "objects", "what is in", "analyze")
    return any(keyword in lowered for keyword in keywords)


class PersonalizationMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        context: AssistantRuntimeContext = request.runtime.context
        memory_lines = [
            f"- [{memory.thread_id}] user={memory.request_message!r} assistant={memory.response_text!r}"
            for memory in context.recent_memories
        ]
        memory_text = "\n".join(memory_lines) if memory_lines else "- No prior cross-session memory recorded."
        uploaded_image_flag = "yes" if context.uploaded_image is not None else "no"

        extra_context = (
            "Resolved hidden context:\n"
            f"- user_id: {context.user_id}\n"
            f"- inferred location: {context.profile.location}\n"
            f"- preferred mode: {context.profile.preferred_mode}\n"
            f"- active mode: {context.mode}\n"
            f"- uploaded_image_available: {uploaded_image_flag}\n"
            f"- preferences: {context.profile.preferences}\n"
            "Mode guidance:\n"
            f"- {_mode_instruction(context.mode)}\n"
            "Cross-session memory:\n"
            f"{memory_text}\n"
            "Operational instructions:\n"
            "- Use current weather tool when the user asks about weather.\n"
            "- Use the image analysis tool when the user asks about the uploaded image.\n"
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": extra_context},
        ]
        return handler(request.override(system_message=SystemMessage(content=new_content)))


class DynamicModelSelectionMiddleware(AgentMiddleware):
    def __init__(self, simple_model, advanced_model) -> None:
        self._simple_model = simple_model
        self._advanced_model = advanced_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        text = _latest_user_text(request.messages)
        context: AssistantRuntimeContext = request.runtime.context

        requires_advanced_model = (
            context.uploaded_image is not None
            or _looks_like_weather_request(text)
            or len(request.messages) > 8
            or len(text) > 400
        )
        selected_model = self._advanced_model if requires_advanced_model else self._simple_model
        context.selected_model = getattr(selected_model, "model_name", None) or str(selected_model)
        return handler(request.override(model=selected_model))


class ToolSelectionMiddleware(AgentMiddleware):
    def __init__(self, tools_by_name: Mapping[str, object]) -> None:
        self._tools_by_name = tools_by_name

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        text = _latest_user_text(request.messages)
        context: AssistantRuntimeContext = request.runtime.context

        selected_tools: list[object] = []
        if _looks_like_weather_request(text):
            selected_tools.append(self._tools_by_name["get_current_weather"])
        if context.uploaded_image is not None and _looks_like_image_request(text):
            selected_tools.append(self._tools_by_name["describe_uploaded_image"])
        if not selected_tools and context.uploaded_image is not None:
            selected_tools.append(self._tools_by_name["describe_uploaded_image"])

        return handler(request.override(tools=selected_tools))


class LangChainAssistantRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        weather_client: WeatherClient,
        image_analyzer: ImageAnalyzer,
        checkpointer,
    ) -> None:
        self._settings = settings
        self._weather_client = weather_client
        self._image_analyzer = image_analyzer
        self._simple_model = self._build_model(settings.simple_model)
        self._advanced_model = self._build_model(settings.advanced_model)
        self._tools = self._build_tools()
        self._agent = create_agent(
            model=self._simple_model,
            tools=list(self._tools.values()),
            system_prompt=BASE_SYSTEM_PROMPT,
            middleware=[
                PersonalizationMiddleware(),
                DynamicModelSelectionMiddleware(self._simple_model, self._advanced_model),
                ToolSelectionMiddleware(self._tools),
            ],
            response_format=self._build_response_format(),
            context_schema=AssistantRuntimeContext,
            checkpointer=checkpointer,
        )

    def respond(
        self,
        message: str,
        runtime_context: AssistantRuntimeContext,
    ) -> AssistantInvocationResult:
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": runtime_context.thread_id}},
            context=runtime_context,
        )

        structured = result.get("structured_response")
        if structured is None:
            final_message = result["messages"][-1]
            payload = AssistantResponsePayload(response=_content_to_text(final_message.content))
        else:
            payload = AssistantResponsePayload.model_validate(structured)

        model_used = runtime_context.selected_model or self._settings.simple_model
        return AssistantInvocationResult(
            payload=payload,
            model_used=model_used,
            tools_invoked=_dedupe(runtime_context.tool_audit),
        )

    def close(self) -> None:
        return None

    def _build_model(self, model_name: str):
        kwargs = {
            "model": model_name,
            "temperature": 0,
            "timeout": self._settings.request_timeout_seconds,
            "max_retries": 2,
            "openrouter_provider": {
                "allow_fallbacks": True,
                "require_parameters": True,
            },
            "app_title": self._settings.openrouter_app_title,
        }
        if self._settings.openrouter_api_key:
            kwargs["api_key"] = self._settings.openrouter_api_key
        if self._settings.openrouter_api_base:
            kwargs["base_url"] = self._settings.openrouter_api_base
        if self._settings.openrouter_app_url:
            kwargs["app_url"] = self._settings.openrouter_app_url
        return ChatOpenRouter(**kwargs)

    def _build_response_format(self):
        if self._settings.structured_output_strategy == "provider":
            return ProviderStrategy(AssistantResponsePayload)
        return ToolStrategy(
            AssistantResponsePayload,
            handle_errors="Return exactly one valid structured response that matches the schema.",
        )

    def _build_tools(self) -> dict[str, object]:
        weather_client = self._weather_client
        image_analyzer = self._image_analyzer

        @tool
        def get_current_weather(
            location: str | None = None,
            *,
            runtime: ToolRuntime[AssistantRuntimeContext],
        ) -> dict:
            """Fetch the current weather. If location is omitted, use the stored user location."""
            resolved_location = location or runtime.context.profile.location
            weather = weather_client.get_current_weather(resolved_location)
            runtime.context.tool_audit.append("get_current_weather")
            runtime.context.weather_result = weather
            return weather.model_dump()

        @tool
        def describe_uploaded_image(
            question: str | None = None,
            *,
            runtime: ToolRuntime[AssistantRuntimeContext],
        ) -> dict:
            """Analyze the uploaded image associated with this request."""
            if runtime.context.uploaded_image is None:
                result = ImageAnalysisResult(
                    description="No uploaded image is available for this request.",
                    objects=[],
                )
            else:
                result = image_analyzer.analyze(runtime.context.uploaded_image, question)
            runtime.context.tool_audit.append("describe_uploaded_image")
            runtime.context.image_result = result
            return result.model_dump()

        return {
            "get_current_weather": get_current_weather,
            "describe_uploaded_image": describe_uploaded_image,
        }


class HeuristicAssistantRunner:
    def __init__(self, weather_client: WeatherClient, image_analyzer: ImageAnalyzer) -> None:
        self._weather_client = weather_client
        self._image_analyzer = image_analyzer

    def respond(
        self,
        message: str,
        runtime_context: AssistantRuntimeContext,
    ) -> AssistantInvocationResult:
        lowered = message.lower()
        payload = AssistantResponsePayload(response="")

        if _looks_like_weather_request(lowered):
            weather = self._weather_client.get_current_weather(runtime_context.profile.location)
            runtime_context.weather_result = weather
            runtime_context.tool_audit.append("get_current_weather")
            payload = AssistantResponsePayload(
                response=self._render_weather_response(weather, runtime_context.mode),
                temperature=weather.temperature,
                summary=weather.summary,
                location=weather.location,
            )
        elif runtime_context.uploaded_image is not None and _looks_like_image_request(lowered):
            image_result = self._image_analyzer.analyze(runtime_context.uploaded_image, message)
            runtime_context.image_result = image_result
            runtime_context.tool_audit.append("describe_uploaded_image")
            payload = AssistantResponsePayload(
                response=self._render_image_response(image_result, runtime_context.mode),
                image_description=image_result.description,
                objects=image_result.objects,
            )
        else:
            payload = AssistantResponsePayload(
                response=self._render_general_response(message, runtime_context),
            )

        runtime_context.selected_model = "heuristic-fallback"
        return AssistantInvocationResult(
            payload=payload,
            model_used="heuristic-fallback",
            tools_invoked=_dedupe(runtime_context.tool_audit),
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _render_weather_response(weather: WeatherResult, mode: str) -> str:
        if mode == "child":
            return (
                f"It is about {weather.temperature} C in {weather.location}, and the weather looks "
                f"{weather.summary}. You may want to dress for that."
            )
        if mode == "expert":
            return (
                f"Current conditions for {weather.location}: {weather.temperature} C with "
                f"{weather.summary}."
            )
        return f"It is {weather.temperature} C and {weather.summary} in {weather.location}."

    @staticmethod
    def _render_image_response(image_result: ImageAnalysisResult, mode: str) -> str:
        if mode == "child":
            return f"Here is what I can tell from the picture: {image_result.description}"
        if mode == "expert":
            return f"Image analysis complete. {image_result.description}"
        return image_result.description

    @staticmethod
    def _render_general_response(message: str, runtime_context: AssistantRuntimeContext) -> str:
        base = (
            f"I used your saved profile for personalization. Your current profile location is "
            f"{runtime_context.profile.location}."
        )
        if runtime_context.mode == "child":
            return f"{base} I am keeping things simple. You said: {message}"
        if runtime_context.mode == "expert":
            return f"{base} Responding in expert mode. Request received: {message}"
        return f"{base} Request received: {message}"
