from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Voice Orchestrator"
    environment: Literal["local", "staging", "production"] = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    llm_model: str = "gpt-4o-mini"
    triage_model: str = "gpt-4o-mini"
    billing_model: str = "gpt-4o-mini"
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_style_prompt: str = (
        "Speak like a calm, precise customer support agent. Keep a helpful tone and short pacing."
    )

    transcription_language: str = "en"
    transcription_prompt: str = (
        "This is a customer support voice call. Preserve account numbers, invoice ids, dates, "
        "amounts, card endings, and filler words only when they change meaning."
    )

    audio_sample_rate_hz: int = 24000
    audio_chunk_ms: int = 250
    speech_start_threshold: float = 900.0
    speech_end_silence_ms: int = 700
    preroll_ms: int = 400
    interrupt_threshold: float = 1200.0
    interrupt_min_chunks: int = 2
    interrupt_preroll_ms: int = 500

    max_recent_messages: int = 8
    max_context_tokens: int = 2200
    max_summary_chars: int = 1200
    max_session_count: int = 250

    db_concurrency: int = 20
    vector_concurrency: int = 30
    openai_concurrency: int = 40
    max_queue_backlog: int = 200
    worker_timeout_seconds: float = 8.0

    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    session_log_dir: Path = Path("logs/sessions")
    redis_url: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.session_log_dir.mkdir(parents=True, exist_ok=True)
    return settings
