from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select

from multimodal_assistant.db import InteractionORM, UserProfileORM
from multimodal_assistant.schemas import (
    AssistantResponsePayload,
    InteractionSummary,
    Mode,
    UserProfile,
)


class UserProfileRepository:
    def __init__(self, session_factory, default_location: str, default_mode: Mode) -> None:
        self._session_factory = session_factory
        self._default_location = default_location
        self._default_mode = default_mode

    def get_or_create(self, user_id: str) -> UserProfile:
        with self._session_factory() as session:
            profile = session.scalar(
                select(UserProfileORM).where(UserProfileORM.user_id == user_id),
            )
            if profile is None:
                profile = UserProfileORM(
                    user_id=user_id,
                    location=self._default_location,
                    preferred_mode=self._default_mode,
                    locale="en-US",
                    preferences={"units": "metric"},
                )
                session.add(profile)
                session.commit()
                session.refresh(profile)
            return self._to_model(profile)

    def update_preferred_mode(self, user_id: str, preferred_mode: Mode) -> None:
        with self._session_factory() as session:
            profile = session.scalar(
                select(UserProfileORM).where(UserProfileORM.user_id == user_id),
            )
            if profile is None:
                profile = UserProfileORM(
                    user_id=user_id,
                    location=self._default_location,
                    preferred_mode=preferred_mode,
                    locale="en-US",
                    preferences={"units": "metric"},
                )
                session.add(profile)
            else:
                profile.preferred_mode = preferred_mode
                profile.updated_at = datetime.now(timezone.utc)
            session.commit()

    def seed_profiles(self, profiles: Iterable[UserProfile]) -> None:
        with self._session_factory() as session:
            for profile_model in profiles:
                existing = session.scalar(
                    select(UserProfileORM).where(UserProfileORM.user_id == profile_model.user_id),
                )
                if existing is None:
                    session.add(
                        UserProfileORM(
                            user_id=profile_model.user_id,
                            location=profile_model.location,
                            preferred_mode=profile_model.preferred_mode,
                            locale=profile_model.locale,
                            preferences=profile_model.preferences,
                        ),
                    )
                    continue
                existing.location = profile_model.location
                existing.preferred_mode = profile_model.preferred_mode
                existing.locale = profile_model.locale
                existing.preferences = profile_model.preferences
                existing.updated_at = datetime.now(timezone.utc)
            session.commit()

    @staticmethod
    def _to_model(profile: UserProfileORM) -> UserProfile:
        return UserProfile(
            user_id=profile.user_id,
            location=profile.location,
            preferred_mode=profile.preferred_mode,  # type: ignore[arg-type]
            locale=profile.locale,
            preferences=profile.preferences or {},
        )


class InteractionRepository:
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def record_interaction(
        self,
        *,
        user_id: str,
        thread_id: str,
        request_message: str,
        response_payload: AssistantResponsePayload,
        tools_invoked: list[str],
        model_used: str,
    ) -> None:
        with self._session_factory() as session:
            session.add(
                InteractionORM(
                    user_id=user_id,
                    thread_id=thread_id,
                    request_message=request_message,
                    response_text=response_payload.response,
                    response_payload=response_payload.model_dump(),
                    tools_invoked=tools_invoked,
                    model_used=model_used,
                ),
            )
            session.commit()

    def list_recent(self, user_id: str, limit: int) -> list[InteractionSummary]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(InteractionORM)
                .where(InteractionORM.user_id == user_id)
                .order_by(InteractionORM.created_at.desc())
                .limit(limit),
            ).all()
            return [
                InteractionSummary(
                    thread_id=row.thread_id,
                    request_message=row.request_message,
                    response_text=row.response_text,
                    created_at=row.created_at,
                )
                for row in rows
            ]
