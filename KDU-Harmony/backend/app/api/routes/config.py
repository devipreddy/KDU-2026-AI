from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()


class PublicConfigResponse(BaseModel):
    project_name: str
    environment: str
    opensearch_configured: bool
    langsmith_tracing_enabled: bool


@router.get("/config", response_model=PublicConfigResponse)
def read_public_config() -> PublicConfigResponse:
    return PublicConfigResponse(
        project_name=settings.project_name,
        environment=settings.app_env,
        opensearch_configured=bool(settings.opensearch_host),
        langsmith_tracing_enabled=settings.langsmith_tracing,
    )
