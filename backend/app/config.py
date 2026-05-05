from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "backend" / ".data"


class Settings(BaseSettings):
    app_name: str = "Travel Booking ChatKit Backend"
    api_prefix: str = "/api"
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    database_path: Path = DATA_DIR / "travel_chatkit.sqlite3"
    jwt_secret: SecretStr = SecretStr(
        "change-me-in-production-with-at-least-32-bytes"
    )
    jwt_algorithm: str = "HS256"
    client_secret_ttl_seconds: int = 30 * 60
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    use_mock_model: bool = False
    human_agent_dashboard_token: SecretStr = SecretStr("agent-demo-token")
    customer_cookie_name: str = "travel_chat_uid"
    session_cookie_name: str = "travel_chat_sid"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _normalize_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
