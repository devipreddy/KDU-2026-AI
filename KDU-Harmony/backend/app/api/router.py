from fastapi import APIRouter

from app.api.routes import audit, auth, config, documents, health, phi, search

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router, tags=["health"])
api_router.include_router(config.router, tags=["config"])
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(phi.router, tags=["phi"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(audit.router, tags=["audit"])
