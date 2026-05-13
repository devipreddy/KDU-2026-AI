from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://healthcare_app:healthcare_app_password@localhost:5432/healthcare_search"
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
    jwt_secret_key: str = Field(
        default="local-development-jwt-secret-change-me",
        alias="JWT_SECRET_KEY",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    jwt_issuer: str = Field(default="healthcare-semantic-search-api", alias="JWT_ISSUER")
    document_storage_root: Path = Field(
        default=Path("../data/storage"), alias="DOCUMENT_STORAGE_ROOT"
    )
    document_storage_key: str = Field(
        default="local-development-document-storage-key",
        alias="DOCUMENT_STORAGE_KEY",
    )
    processed_text_root: Path = Field(
        default=Path("../data/processed"),
        alias="PROCESSED_TEXT_ROOT",
    )
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")
    ocr_confidence_review_threshold: float = Field(
        default=0.80,
        alias="OCR_CONFIDENCE_REVIEW_THRESHOLD",
    )
    ocr_render_dpi: int = Field(default=200, alias="OCR_RENDER_DPI")
    tesseract_cmd: str | None = Field(default=None, alias="TESSERACT_CMD")
    chroma_host: AnyHttpUrl | None = Field(default=None, alias="CHROMA_HOST")
    chroma_collection: str = Field(default="medical_record_chunks", alias="CHROMA_COLLECTION")
    chroma_persist_path: Path = Field(
        default=Path("../data/chroma"),
        alias="CHROMA_PERSIST_PATH",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
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
