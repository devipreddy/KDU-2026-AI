from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.agents.billing import BillingAgent
from app.agents.consensus import ConsensusAgent
from app.agents.db_agent import DBAgent
from app.agents.triage import TriageAgent
from app.agents.vector_agent import VectorAgent
from app.config import Settings, get_settings
from app.core.logging import setup_logging
from app.core.models import SessionPhase, SessionSnapshot, TextTurnRequest
from app.services.audio import AudioTurnBuffer
from app.services.concurrency import ConcurrencyQueue
from app.services.monitoring import Metrics
from app.services.openai_provider import OpenAIProvider
from app.services.orchestrator import Coordinator
from app.services.pruning import ConversationPruner
from app.services.replay import SessionReplayStore
from app.services.security import SecurityGuard
from app.services.session_manager import VoiceSessionManager
from app.storage.repositories import AccountRepository, VectorRepository

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    provider: OpenAIProvider
    replay_store: SessionReplayStore
    metrics: Metrics
    session_manager: VoiceSessionManager


def build_services(settings: Settings) -> ServiceContainer:
    provider = OpenAIProvider(settings)
    pruner = ConversationPruner(
        max_recent_messages=settings.max_recent_messages,
        max_context_tokens=settings.max_context_tokens,
        max_summary_chars=settings.max_summary_chars,
    )
    security = SecurityGuard()
    db_repository = AccountRepository(settings.data_dir / "mock_accounts.json")
    vector_repository = VectorRepository(settings.data_dir / "knowledge_base.json")

    triage_agent = TriageAgent(provider=provider, model=settings.triage_model)
    billing_agent = BillingAgent(provider=provider, model=settings.billing_model)
    db_agent = DBAgent(repository=db_repository)
    vector_agent = VectorAgent(repository=vector_repository)
    consensus_agent = ConsensusAgent()

    coordinator = Coordinator(
        triage_agent=triage_agent,
        billing_agent=billing_agent,
        db_agent=db_agent,
        vector_agent=vector_agent,
        consensus_agent=consensus_agent,
        pruner=pruner,
        security=security,
        db_queue=ConcurrencyQueue(settings.db_concurrency, settings.max_queue_backlog),
        vector_queue=ConcurrencyQueue(settings.vector_concurrency, settings.max_queue_backlog),
        worker_timeout_seconds=settings.worker_timeout_seconds,
    )

    replay_store = SessionReplayStore(settings.session_log_dir)
    metrics = Metrics()

    def audio_buffer_factory() -> AudioTurnBuffer:
        return AudioTurnBuffer(
            sample_rate_hz=settings.audio_sample_rate_hz,
            chunk_ms=settings.audio_chunk_ms,
            speech_start_threshold=settings.speech_start_threshold,
            speech_end_silence_ms=settings.speech_end_silence_ms,
            preroll_ms=settings.preroll_ms,
            interrupt_threshold=settings.interrupt_threshold,
            interrupt_min_chunks=settings.interrupt_min_chunks,
            interrupt_preroll_ms=settings.interrupt_preroll_ms,
        )

    session_manager = VoiceSessionManager(
        coordinator=coordinator,
        provider=provider,
        replay_store=replay_store,
        metrics=metrics,
        audio_buffer_factory=audio_buffer_factory,
    )

    return ServiceContainer(
        settings=settings,
        provider=provider,
        replay_store=replay_store,
        metrics=metrics,
        session_manager=session_manager,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging()
    services = build_services(settings)
    app.state.services = services
    logger.info("application_started")
    try:
        yield
    finally:
        await services.provider.close()
        logger.info("application_stopped")


app = FastAPI(title="Voice Orchestrator", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    services: ServiceContainer = app.state.services
    return {
        "status": "ok",
        "environment": services.settings.environment,
        "openai_configured": bool(services.settings.openai_api_key),
        "openrouter_configured": bool(services.settings.openrouter_api_key),
        "active_sessions": len(services.session_manager.sessions),
    }


@app.get("/metrics")
async def metrics() -> Response:
    services: ServiceContainer = app.state.services
    payload, content_type = services.metrics.render()
    return Response(payload, media_type=content_type)


@app.get("/api/sessions/{session_id}")
async def get_session_snapshot(session_id: str) -> JSONResponse:
    services: ServiceContainer = app.state.services
    snapshot = services.session_manager.get_snapshot(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(snapshot)


@app.get("/api/replay/{session_id}")
async def get_session_replay(session_id: str) -> JSONResponse:
    services: ServiceContainer = app.state.services
    return JSONResponse({"events": services.replay_store.read(session_id)})


@app.post("/api/demo/text-turn")
async def simulate_text_turn(request: TextTurnRequest) -> JSONResponse:
    services: ServiceContainer = app.state.services
    snapshot = SessionSnapshot(
        session_id=request.session_id or f"sim_{uuid4().hex}",
        user_id=request.user_id,
        trace_id=request.trace_id or f"trace_{uuid4().hex}",
        phase=SessionPhase.PROCESSING,
    )
    outcome = await services.session_manager.coordinator.handle_turn(snapshot, request.transcript)
    return JSONResponse(
        {
            "session": outcome["snapshot"].model_dump(mode="json"),
            "handoff": outcome["handoff"].model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in outcome["observations"]],
            "worker_results": [item.model_dump(mode="json") for item in outcome["worker_results"]],
            "consensus": (
                outcome["consensus"].model_dump(mode="json")
                if outcome["consensus"] is not None
                else None
            ),
            "assistant_text": outcome["assistant_text"],
        }
    )


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket) -> None:
    services: ServiceContainer = app.state.services
    user_id = websocket.query_params.get("user_id", "demo-user")
    session = await services.session_manager.connect(websocket, user_id)
    try:
        while True:
            payload = await websocket.receive_json()
            await session.handle_message(payload)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    finally:
        await services.session_manager.disconnect(session)
