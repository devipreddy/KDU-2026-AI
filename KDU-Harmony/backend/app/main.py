from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    logger.info("application_starting", app=settings.project_name, environment=settings.app_env)
    yield
    logger.info("application_stopping", app=settings.project_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="HIPAA-aware semantic search over synthetic medical records.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/health")
    def read_root_health() -> dict[str, str]:
        return {"status": "ok", "service": "healthcare-semantic-search-api"}

    return app


app = create_app()
