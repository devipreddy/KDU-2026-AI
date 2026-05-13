from fastapi import APIRouter

from app.api.routes import config, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(config.router, tags=["config"])
