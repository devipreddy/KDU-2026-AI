from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    app_name: str = "multimodal-context-assistant"
    environment: str = "development"
    database_url: str = f"sqlite:///{(DEFAULT_DATA_DIR / 'assistant.db').as_posix()}"
    checkpoint_db_path: str = str(DEFAULT_DATA_DIR / "checkpoints.sqlite")
    default_location: str = "San Francisco, US"
    default_mode: Literal["default", "expert", "child"] = "default"
    profile_preview_limit: int = 3
    seed_demo_profiles: bool = True
    weather_provider: Literal["mock", "open-meteo"] = "mock"
    weather_timeout_seconds: float = 8.0
    request_timeout_seconds: float = 60.0
    openrouter_api_key: str | None = None
    openrouter_api_base: str | None = None
    openrouter_app_title: str = "Multimodal Context Assistant"
    openrouter_app_url: str | None = None
    simple_model: str = "qwen/qwen3.6-plus:free"
    advanced_model: str = "qwen/qwen3.6-plus:free"
    vision_model: str = "qwen/qwen3.6-plus:free"
    structured_output_strategy: Literal["tool", "provider"] = "tool"
    enable_offline_fallback: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ASSISTANT_",
        extra="ignore",
    )

    @property
    def data_dir(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            db_path = Path(self.database_url.removeprefix("sqlite:///"))
            return db_path.parent
        return Path(self.checkpoint_db_path).resolve().parent

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_db_path).expanduser().resolve().parent.mkdir(
            parents=True,
            exist_ok=True,
        )
