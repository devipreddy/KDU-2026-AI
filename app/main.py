from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_job_queue_service
from app.api.routes.costs import router as costs_router
from app.api.routes.files import router as files_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import init_db

settings = get_settings()
configure_logging(settings.debug)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(files_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)
app.include_router(costs_router, prefix=settings.api_prefix)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    get_job_queue_service().startup()


@app.on_event("shutdown")
def on_shutdown() -> None:
    get_job_queue_service().shutdown()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
