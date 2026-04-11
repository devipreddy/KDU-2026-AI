from pathlib import Path

from multimodal_assistant.bootstrap import build_container
from multimodal_assistant.config import Settings
from multimodal_assistant.db import create_session_factory, create_sqlalchemy_engine
from multimodal_assistant.repositories import InteractionRepository, UserProfileRepository
from multimodal_assistant.schemas import ChatRequest


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'assistant.db').as_posix()}",
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        default_location="Seattle, US",
        seed_demo_profiles=False,
        openrouter_api_key=None,
        enable_offline_fallback=True,
    )


def test_service_persists_profiles_and_interactions(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    container = build_container(settings)

    try:
        weather_response = container.service.handle_chat(
            ChatRequest(
                user_id="alice",
                thread_id="thread-weather",
                message="What's the weather today?",
            ),
        )
        general_response = container.service.handle_chat(
            ChatRequest(
                user_id="alice",
                thread_id="thread-style",
                message="Keep helping me like a child.",
                mode="child",
            ),
        )
    finally:
        container.close()

    engine = create_sqlalchemy_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    profile_repository = UserProfileRepository(
        session_factory=session_factory,
        default_location=settings.default_location,
        default_mode=settings.default_mode,
    )
    interaction_repository = InteractionRepository(session_factory=session_factory)

    profile = profile_repository.get_or_create("alice")
    interactions = interaction_repository.list_recent("alice", limit=10)
    engine.dispose()

    assert weather_response.data.location == "Seattle, US"
    assert weather_response.metadata.tools_invoked == ["get_current_weather"]
    assert general_response.metadata.thread_id == "thread-style"
    assert profile.preferred_mode == "child"
    assert len(interactions) == 2
