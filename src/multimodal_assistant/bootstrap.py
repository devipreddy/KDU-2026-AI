from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.memory import InMemorySaver

from multimodal_assistant.assistant import HeuristicAssistantRunner, LangChainAssistantRunner
from multimodal_assistant.config import Settings
from multimodal_assistant.db import create_session_factory, create_sqlalchemy_engine, init_db
from multimodal_assistant.image_analysis import (
    FallbackImageAnalyzer,
    LangChainVisionAnalyzer,
    MetadataImageAnalyzer,
)
from multimodal_assistant.input_processing import InputProcessor
from multimodal_assistant.repositories import InteractionRepository, UserProfileRepository
from multimodal_assistant.schemas import UserProfile
from multimodal_assistant.service import AssistantService
from multimodal_assistant.weather import MockWeatherClient, OpenMeteoWeatherClient
from langchain_openrouter import ChatOpenRouter


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    service: AssistantService
    engine: object
    weather_client: object
    image_analyzer: object
    runner: object

    def close(self) -> None:
        self.runner.close()
        self.weather_client.close()
        self.image_analyzer.close()
        self.engine.dispose()


def build_container(settings: Settings | None = None) -> ApplicationContainer:
    active_settings = settings or Settings()
    active_settings.ensure_directories()

    engine = create_sqlalchemy_engine(active_settings.database_url)
    init_db(engine)
    session_factory = create_session_factory(engine)
    profile_repository = UserProfileRepository(
        session_factory=session_factory,
        default_location=active_settings.default_location,
        default_mode=active_settings.default_mode,
    )
    interaction_repository = InteractionRepository(session_factory=session_factory)

    if active_settings.seed_demo_profiles:
        profile_repository.seed_profiles(
            [
                UserProfile(
                    user_id="user_123",
                    location="San Francisco, US",
                    preferred_mode="expert",
                    preferences={"units": "metric", "tone": "technical"},
                ),
                UserProfile(
                    user_id="kid_001",
                    location="Bengaluru, IN",
                    preferred_mode="child",
                    preferences={"units": "metric", "tone": "simple"},
                ),
            ],
        )

    weather_client = _build_weather_client(active_settings)
    image_analyzer = _build_image_analyzer(active_settings)
    runner = _build_runner(
        settings=active_settings,
        weather_client=weather_client,
        image_analyzer=image_analyzer,
    )

    service = AssistantService(
        settings=active_settings,
        profile_repository=profile_repository,
        interaction_repository=interaction_repository,
        input_processor=InputProcessor(),
        runner=runner,
    )

    return ApplicationContainer(
        settings=active_settings,
        service=service,
        engine=engine,
        weather_client=weather_client,
        image_analyzer=image_analyzer,
        runner=runner,
    )


def _build_weather_client(settings: Settings):
    if settings.weather_provider == "open-meteo":
        return OpenMeteoWeatherClient(settings.weather_timeout_seconds)
    return MockWeatherClient()


def _build_image_analyzer(settings: Settings):
    metadata_fallback = MetadataImageAnalyzer()
    if not settings.openrouter_api_key:
        return FallbackImageAnalyzer(primary=None, fallback=metadata_fallback)

    vision_model_kwargs = {
        "model": settings.vision_model,
        "temperature": 0,
        "timeout": settings.request_timeout_seconds,
        "max_retries": 2,
        "api_key": settings.openrouter_api_key,
        "openrouter_provider": {
            "allow_fallbacks": True,
            "require_parameters": True,
        },
        "app_title": settings.openrouter_app_title,
    }
    if settings.openrouter_api_base:
        vision_model_kwargs["base_url"] = settings.openrouter_api_base
    if settings.openrouter_app_url:
        vision_model_kwargs["app_url"] = settings.openrouter_app_url

    primary = LangChainVisionAnalyzer(ChatOpenRouter(**vision_model_kwargs))
    return FallbackImageAnalyzer(primary=primary, fallback=metadata_fallback)


def _build_runner(*, settings: Settings, weather_client, image_analyzer):
    if not settings.openrouter_api_key and settings.enable_offline_fallback:
        return HeuristicAssistantRunner(weather_client, image_analyzer)

    checkpointer = InMemorySaver()
    runner = LangChainAssistantRunner(
        settings=settings,
        weather_client=weather_client,
        image_analyzer=image_analyzer,
        checkpointer=checkpointer,
    )
    return runner
