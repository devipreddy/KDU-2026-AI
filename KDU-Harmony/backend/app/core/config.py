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
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_ocr_model: str = Field(
        default="baidu/qianfan-ocr-fast",
        alias="OPENROUTER_OCR_MODEL",
    )
    openrouter_ocr_fallback_models: str = Field(
        default="baidu/qianfan-ocr-fast:free",
        alias="OPENROUTER_OCR_FALLBACK_MODELS",
    )
    openrouter_ocr_timeout_seconds: float = Field(
        default=120.0,
        alias="OPENROUTER_OCR_TIMEOUT_SECONDS",
    )
    openrouter_ocr_max_tokens: int = Field(default=4096, alias="OPENROUTER_OCR_MAX_TOKENS")
    openrouter_ocr_max_workers: int = Field(default=2, alias="OPENROUTER_OCR_MAX_WORKERS")
    presidio_enabled: bool = Field(default=True, alias="PRESIDIO_ENABLED")
    presidio_score_threshold: float = Field(default=0.65, alias="PRESIDIO_SCORE_THRESHOLD")
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
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        alias="RERANKER_MODEL_NAME",
    )
    reranker_fallback_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_FALLBACK_MODEL_NAME",
    )
    reranker_batch_size: int = Field(default=16, alias="RERANKER_BATCH_SIZE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_timeout_seconds: float = Field(default=30.0, alias="OPENAI_TIMEOUT_SECONDS")
    openai_max_output_tokens: int = Field(default=600, alias="OPENAI_MAX_OUTPUT_TOKENS")
    index_on_ingestion: bool = Field(default=False, alias="INDEX_ON_INGESTION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    otel_enabled: bool = Field(default=False, alias="OTEL_ENABLED")
    otel_service_name: str | None = Field(default=None, alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_exporter_otlp_insecure: bool = Field(
        default=True,
        alias="OTEL_EXPORTER_OTLP_INSECURE",
    )
    otel_sqlalchemy_instrumentation: bool = Field(
        default=True,
        alias="OTEL_SQLALCHEMY_INSTRUMENTATION",
    )
    otel_excluded_urls: str = Field(default="/health,/api/v1/health", alias="OTEL_EXCLUDED_URLS")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="healthcare-semantic-search-local",
        alias="LANGSMITH_PROJECT",
    )
    langsmith_endpoint: str | None = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )
    encryption_key_id: str = Field(default="local-development-key", alias="ENCRYPTION_KEY_ID")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
