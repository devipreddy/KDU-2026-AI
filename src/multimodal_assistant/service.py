from __future__ import annotations

from datetime import datetime, timezone

from multimodal_assistant.assistant import AssistantRunner
from multimodal_assistant.config import Settings
from multimodal_assistant.input_processing import InputProcessor
from multimodal_assistant.repositories import InteractionRepository, UserProfileRepository
from multimodal_assistant.schemas import (
    AssistantResponsePayload,
    AssistantRuntimeContext,
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
)


class AssistantService:
    def __init__(
        self,
        *,
        settings: Settings,
        profile_repository: UserProfileRepository,
        interaction_repository: InteractionRepository,
        input_processor: InputProcessor,
        runner: AssistantRunner,
    ) -> None:
        self._settings = settings
        self._profile_repository = profile_repository
        self._interaction_repository = interaction_repository
        self._input_processor = input_processor
        self._runner = runner

    def handle_chat(self, chat_request: ChatRequest) -> ChatResponse:
        profile = self._profile_repository.get_or_create(chat_request.user_id)
        resolved_mode = (
            profile.preferred_mode
            if chat_request.mode == "default"
            else chat_request.mode
        )
        normalized_image = self._input_processor.normalize_image(chat_request.image)
        normalized_message = self._input_processor.normalize_message(
            chat_request.message,
            normalized_image is not None,
        )
        recent_memories = self._interaction_repository.list_recent(
            chat_request.user_id,
            self._settings.profile_preview_limit,
        )

        runtime_context = AssistantRuntimeContext(
            user_id=chat_request.user_id,
            thread_id=chat_request.thread_id,
            mode=resolved_mode,
            profile=profile,
            recent_memories=recent_memories,
            uploaded_image=normalized_image,
        )

        invocation = self._runner.respond(normalized_message, runtime_context)
        payload = self._merge_tool_results(invocation.payload, runtime_context)

        if chat_request.mode != "default":
            self._profile_repository.update_preferred_mode(chat_request.user_id, resolved_mode)

        self._interaction_repository.record_interaction(
            user_id=chat_request.user_id,
            thread_id=chat_request.thread_id,
            request_message=normalized_message,
            response_payload=payload,
            tools_invoked=invocation.tools_invoked,
            model_used=invocation.model_used,
        )

        return ChatResponse(
            data=payload,
            metadata=ChatResponseMetadata(
                model_used=invocation.model_used,
                tools_invoked=invocation.tools_invoked,
                thread_id=chat_request.thread_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )

    @staticmethod
    def _merge_tool_results(
        payload: AssistantResponsePayload,
        runtime_context: AssistantRuntimeContext,
    ) -> AssistantResponsePayload:
        updates: dict = {}
        if runtime_context.weather_result is not None:
            if payload.temperature is None:
                updates["temperature"] = runtime_context.weather_result.temperature
            if payload.summary is None:
                updates["summary"] = runtime_context.weather_result.summary
            if payload.location is None:
                updates["location"] = runtime_context.weather_result.location
        if runtime_context.image_result is not None:
            if payload.image_description is None:
                updates["image_description"] = runtime_context.image_result.description
            if not payload.objects:
                updates["objects"] = runtime_context.image_result.objects
        return payload.model_copy(update=updates)
