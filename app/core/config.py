from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Content Accessibility Suite"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    app_api_key: str | None = None

    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str | None = "http://localhost:8501"
    openrouter_app_title: str = "Content Accessibility Suite"

    openai_generation_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openrouter_generation_model: str = "openai/gpt-4o-mini"
    openrouter_vision_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str | None = "openai/text-embedding-3-small"

    whisper_model_id: str = "openai/whisper-small"
    whisper_device: str = "cpu"
    whisper_torch_dtype: str = "float32"
    whisper_chunk_length_s: int = 30
    whisper_batch_size: int = 8
    huggingface_token: str | None = None
    speaker_diarization_enabled: bool = True
    speaker_diarization_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_diarization_distance_threshold: float = 0.35
    speaker_diarization_min_segment_seconds: float = 1.0

    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")
    database_url: str = "sqlite:///./data/app.db"
    processing_max_workers: int = 2

    max_upload_size_mb: int = 50
    vision_detail: str = "low"
    pdf_text_threshold_chars: int = 120
    pdf_image_trigger_count: int = 2

    summary_max_input_tokens: int = 6000
    summary_chunk_tokens: int = 3000
    summary_chunk_overlap_tokens: int = 300
    embedding_chunk_tokens: int = 700
    embedding_chunk_overlap_tokens: int = 120

    search_default_limit: int = 5
    search_max_limit: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def ensure_directories(self) -> "Settings":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
