from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class PublicConfigResponse(BaseModel):
    project_name: str
    environment: str
    chroma_configured: bool
    chroma_collection: str
    opentelemetry_enabled: bool
    langsmith_tracing_enabled: bool
    observability_redaction: str


@router.get("/config", response_model=PublicConfigResponse)
def read_public_config() -> PublicConfigResponse:
    return PublicConfigResponse(
        project_name=settings.project_name,
        environment=settings.app_env,
        chroma_configured=bool(settings.chroma_host),
        chroma_collection=settings.chroma_collection,
        opentelemetry_enabled=settings.otel_enabled,
        langsmith_tracing_enabled=settings.langsmith_tracing,
        observability_redaction="enabled",
    )
