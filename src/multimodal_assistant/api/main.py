from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from multimodal_assistant.bootstrap import ApplicationContainer, build_container
from multimodal_assistant.config import Settings
from multimodal_assistant.schemas import ChatRequest, ChatResponse


def create_app(
    *,
    settings: Settings | None = None,
    container: ApplicationContainer | None = None,
) -> FastAPI:
    active_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_container = container or build_container(active_settings)
        app.state.container = active_container
        try:
            yield
        finally:
            active_container.close()

    application = FastAPI(
        title=active_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/v1/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatRequest, http_request: Request) -> ChatResponse:
        try:
            return http_request.app.state.container.service.handle_chat(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    return application


app = create_app()


def run() -> None:
    settings = Settings()
    uvicorn.run(
        "multimodal_assistant.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
