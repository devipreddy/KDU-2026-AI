from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.observability import configure_observability, log_observability_status

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log_observability_status(app.state.observability_status)
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

    @app.middleware("http")
    async def request_observability_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        bind_contextvars(request_id=request_id)
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((perf_counter() - started_at) * 1000, 3)
            logger.exception(
                "http_request_failed",
                http={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            clear_contextvars()
            raise

        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http_request_completed",
            http={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        clear_contextvars()
        return response

    @app.get("/health")
    def read_root_health() -> dict[str, str]:
        return {"status": "ok", "service": "healthcare-semantic-search-api"}

    app.state.observability_status = configure_observability(app)
    return app


app = create_app()
