from __future__ import annotations

from typing import Annotated, Any, Literal

from chatkit.actions import Action
from pydantic import BaseModel, Field, TypeAdapter


class BookNowClientPayload(BaseModel):
    offer_id: str


class BookNowServerPayload(BaseModel):
    offer_id: str
    idempotency_key: str


class RequestHandoffPayload(BaseModel):
    reason: str | None = None


class ResumeAIPayload(BaseModel):
    note: str | None = None


class BookNowClientAction(Action[Literal["travel.book_now"], BookNowClientPayload]):
    pass


class BookNowServerAction(
    Action[Literal["travel.book_now.confirm"], BookNowServerPayload]
):
    pass


class RequestHandoffAction(
    Action[Literal["support.request_handoff"], RequestHandoffPayload]
):
    pass


class ResumeAIAction(Action[Literal["support.resume_ai"], ResumeAIPayload]):
    pass


AppAction = Annotated[
    BookNowServerAction | RequestHandoffAction | ResumeAIAction,
    Field(discriminator="type"),
]

APP_ACTION_ADAPTER = TypeAdapter(AppAction)


def parse_app_action(action: Action[str, Any]) -> AppAction:
    return APP_ACTION_ADAPTER.validate_python(action.model_dump())

