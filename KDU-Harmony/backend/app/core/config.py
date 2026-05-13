from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://healthcare_app:healthcare_app_password"
    "@localhost:5432/healthcare_search"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    project_name: str = Field(default="Healthcare Semantic Search API", alias="PROJECT_NAME")
    backend_cors_origins: str = Field(default="http://localhost:5173", alias="BACKEND_CORS_ORIGINS")
    database_url: str = Field(default=DEFAULT_DATABASE_URL, alias="DATABASE_URL")
    opensearch_host: AnyHttpUrl | None = Field(default=None, alias="OPENSEARCH_HOST")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    encryption_key_id: str = Field(default="local-development-key", alias="ENCRYPTION_KEY_ID")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
