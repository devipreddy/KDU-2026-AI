from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.schemas import (
    CacheInvalidateRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
)
from app.services.assistant import AssistantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    service = AssistantService(settings)
    app.state.settings = settings
    app.state.service = service
    app.state.started_at = datetime.now(timezone.utc)
    try:
        yield
    finally:
        await service.close()


settings: Settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> AssistantService:
    return app.state.service


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        """
        <html>
          <head>
            <title>Multi-Function AI Assistant API</title>
            <style>
              body {
                font-family: Arial, sans-serif;
                background: #f7f3eb;
                color: #1f2430;
                margin: 0;
                padding: 48px 24px;
              }
              .card {
                max-width: 760px;
                margin: 0 auto;
                background: #fffaf2;
                border: 1px solid rgba(31, 36, 48, 0.08);
                border-radius: 24px;
                padding: 32px;
                box-shadow: 0 24px 60px rgba(44, 37, 28, 0.1);
              }
              h1 {
                margin-top: 0;
              }
              code {
                background: rgba(15, 118, 110, 0.08);
                padding: 2px 6px;
                border-radius: 8px;
              }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>Multi-Function AI Assistant API</h1>
              <p>This process is the backend service.</p>
              <p>Run the Streamlit UI with <code>streamlit run streamlit_app.py</code>.</p>
              <p>API endpoints remain available at <code>/v1/chat</code>, <code>/v1/chat/stream</code>, <code>/v1/health</code>, and <code>/v1/metrics</code>.</p>
            </div>
          </body>
        </html>
        """
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await get_service().chat(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    stream = get_service().stream_chat(request)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    started_at = app.state.started_at
    now = datetime.now(timezone.utc)
    uptime_seconds = int((now - started_at).total_seconds())
    return HealthResponse(
        status="healthy",
        uptime=f"{uptime_seconds}s",
        dependencies=get_service().get_dependency_status(),
    )


@app.get("/v1/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(**get_service().get_metrics())


@app.get("/v1/circuit-breakers")
async def circuit_breakers() -> dict[str, dict[str, object]]:
    return get_service().get_circuit_breaker_status()


@app.post("/v1/cache/invalidate")
async def invalidate_cache(request: CacheInvalidateRequest) -> dict[str, object]:
    invalidated = get_service().invalidate_cache(request.key)
    return {"success": invalidated, "key": request.key}
