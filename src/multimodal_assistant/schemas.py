from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Mode = Literal["default", "expert", "child"]


class ImagePayload(BaseModel):
    data: str = Field(
        ...,
        description="Base64 encoded image bytes or a full data URL.",
        min_length=1,
    )
    mime_type: str | None = Field(
        default=None,
        description="Optional MIME type. Inferred when omitted.",
    )


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(default="", description="User text input.")
    image: str | ImagePayload | None = Field(
        default=None,
        description="Optional uploaded image, either as a raw base64 string or an object.",
    )
    mode: Mode = "default"
    thread_id: str = Field(default="default", min_length=1)

    @model_validator(mode="after")
    def validate_has_user_input(self) -> "ChatRequest":
        if not self.message.strip() and self.image is None:
            raise ValueError("Either message or image must be provided.")
        return self


class UserProfile(BaseModel):
    user_id: str
    location: str
    preferred_mode: Mode = "default"
    locale: str = "en-US"
    preferences: dict[str, Any] = Field(default_factory=dict)


class InteractionSummary(BaseModel):
    thread_id: str
    request_message: str
    response_text: str
    created_at: datetime


class WeatherResult(BaseModel):
    temperature: float
    summary: str
    location: str


class ImageAnalysisResult(BaseModel):
    description: str
    objects: list[str] = Field(default_factory=list)


class AssistantResponsePayload(BaseModel):
    response: str = Field(description="Primary assistant response text.")
    temperature: float | None = Field(
        default=None,
        description="Temperature if the response is weather related; otherwise null.",
    )
    summary: str | None = Field(
        default=None,
        description="Weather summary if applicable; otherwise null.",
    )
    location: str | None = Field(
        default=None,
        description="Resolved location if applicable; otherwise null.",
    )
    image_description: str | None = Field(
        default=None,
        description="Image description when an image is analyzed; otherwise null.",
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Objects detected in the uploaded image, or an empty list.",
    )


class ChatResponseMetadata(BaseModel):
    model_used: str
    tools_invoked: list[str] = Field(default_factory=list)
    thread_id: str
    timestamp: str


class ChatResponse(BaseModel):
    status: Literal["success"] = "success"
    data: AssistantResponsePayload
    metadata: ChatResponseMetadata


@dataclass(slots=True)
class NormalizedImage:
    base64_data: str
    mime_type: str
    raw_bytes: bytes
    size_bytes: int


@dataclass(slots=True)
class AssistantRuntimeContext:
    user_id: str
    thread_id: str
    mode: Mode
    profile: UserProfile
    recent_memories: list[InteractionSummary] = field(default_factory=list)
    uploaded_image: NormalizedImage | None = None
    tool_audit: list[str] = field(default_factory=list)
    selected_model: str | None = None
    weather_result: WeatherResult | None = None
    image_result: ImageAnalysisResult | None = None


@dataclass(slots=True)
class AssistantInvocationResult:
    payload: AssistantResponsePayload
    model_used: str
    tools_invoked: list[str]
