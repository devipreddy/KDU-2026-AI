from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from chatkit.server import NonStreamingResult, StreamingResult
from chatkit.store import NotFoundError
from fastapi import FastAPI, Header, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthError, AuthService
from .catalog import build_catalog
from .chat_server import TravelChatKitServer
from .config import Settings, get_settings
from .context import RequestContext
from .realtime import RealtimeHub
from .schemas import (
    ClaimHandoffRequest,
    HumanMessageRequest,
    ReleaseHandoffRequest,
    SessionResponse,
    ThreadSummaryResponse,
)
from .store import SQLiteChatStore


settings = get_settings()
auth_service = AuthService(settings)
store = SQLiteChatStore(settings.database_path)
hub = RealtimeHub()
chat_server = TravelChatKitServer(
    settings=settings,
    store=store,
    catalog=build_catalog(),
    hub=hub,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
    await store.initialize()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthError)
async def handle_auth_error(_: Request, exc: AuthError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(NotFoundError)
async def handle_not_found(_: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{settings.api_prefix}/session", response_model=SessionResponse)
async def create_customer_session(request: Request, response: Response) -> SessionResponse:
    identity = auth_service.ensure_customer_identity(request, response)
    session = auth_service.issue_client_secret(
        user_id=identity.user_id,
        session_id=identity.session_id,
        role="customer",
    )
    return SessionResponse.model_validate(session.model_dump())


@app.post(f"{settings.api_prefix}/agent/session", response_model=SessionResponse)
async def create_agent_session(
    request: Request,
    response: Response,
    x_agent_token: str | None = Header(default=None),
) -> SessionResponse:
    auth_service.validate_agent_dashboard_token(x_agent_token)
    identity = auth_service.ensure_customer_identity(request, response)
    session = auth_service.issue_client_secret(
        user_id=f"agent_{identity.user_id}",
        session_id=identity.session_id,
        role="agent",
        display_name="Travel Specialist",
    )
    return SessionResponse.model_validate(session.model_dump())


@app.post(f"{settings.api_prefix}/chatkit", response_model=None)
async def chatkit_endpoint(request: Request) -> Response:
    context = auth_service.require_request_context(request)
    body = await request.body()
    await pre_authorize_chatkit_request(body, context)
    result = await chat_server.process(body, context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if isinstance(result, NonStreamingResult):
        return Response(content=result.json, media_type="application/json")
    raise RuntimeError("Unexpected ChatKit result type")


@app.get(
    f"{settings.api_prefix}/threads/{{thread_id}}/summary",
    response_model=ThreadSummaryResponse,
)
async def get_thread_summary(thread_id: str, request: Request) -> ThreadSummaryResponse:
    context = auth_service.require_request_context(request)
    summary = await chat_server.thread_summary(thread_id=thread_id, context=context)
    return ThreadSummaryResponse.model_validate(summary)


@app.get(f"{settings.api_prefix}/agent/handoff/queue")
async def handoff_queue(request: Request) -> list[ThreadSummaryResponse]:
    context = auth_service.require_request_context(request)
    if context.role != "agent":
        raise AuthError("Agent role is required", 403)
    page = await store.load_threads(limit=200, after=None, order="desc", context=context)
    summaries: list[ThreadSummaryResponse] = []
    for thread in page.data:
        mode = str(thread.metadata.get("conversation_mode", "ai"))
        if mode == "ai":
            continue
        summary = await chat_server.thread_summary(thread_id=thread.id, context=context)
        summaries.append(ThreadSummaryResponse.model_validate(summary))
    return summaries


@app.get(f"{settings.api_prefix}/agent/threads/{{thread_id}}")
async def get_agent_thread(thread_id: str, request: Request) -> dict[str, Any]:
    context = auth_service.require_request_context(request)
    if context.role != "agent":
        raise AuthError("Agent role is required", 403)
    thread = await store.load_thread(thread_id, context)
    items = await store.load_all_thread_items(thread_id, context)
    return {
        "thread": thread.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in items],
    }


@app.post(f"{settings.api_prefix}/agent/handoff/{{thread_id}}/claim")
async def claim_handoff(
    thread_id: str,
    payload: ClaimHandoffRequest,
    request: Request,
) -> dict[str, str]:
    context = auth_service.require_request_context(request)
    if context.role != "agent":
        raise AuthError("Agent role is required", 403)
    await chat_server.claim_handoff(
        thread_id=thread_id,
        agent_name=payload.display_name,
        context=context,
    )
    return {"status": "claimed"}


@app.post(f"{settings.api_prefix}/agent/handoff/{{thread_id}}/message")
async def send_human_message(
    thread_id: str,
    payload: HumanMessageRequest,
    request: Request,
) -> dict[str, Any]:
    context = auth_service.require_request_context(request)
    if context.role != "agent":
        raise AuthError("Agent role is required", 403)
    item = await chat_server.post_human_message(
        thread_id=thread_id,
        text=payload.text,
        context=context,
    )
    return {"status": "sent", "item": item.model_dump(mode="json")}


@app.post(f"{settings.api_prefix}/agent/handoff/{{thread_id}}/release")
async def release_handoff(
    thread_id: str,
    payload: ReleaseHandoffRequest,
    request: Request,
) -> dict[str, str]:
    context = auth_service.require_request_context(request)
    if context.role != "agent":
        raise AuthError("Agent role is required", 403)
    await chat_server.release_handoff(
        thread_id=thread_id,
        resume_ai=payload.resume_ai,
        context=context,
    )
    return {"status": "released"}


@app.websocket("/ws/threads/{thread_id}")
async def thread_updates(websocket: WebSocket, thread_id: str) -> None:
    try:
        context = await auth_service.require_websocket_context(websocket)
        await store.load_thread(thread_id, context)
    except AuthError:
        await websocket.close(code=4401)
        return
    except NotFoundError:
        await websocket.close(code=4404)
        return

    await hub.connect_thread(thread_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect_thread(thread_id, websocket)


@app.websocket("/ws/handoff/queue")
async def handoff_updates(websocket: WebSocket) -> None:
    try:
        context = await auth_service.require_websocket_context(
            websocket,
            expected_role="agent",
        )
        if context.role != "agent":
            raise AuthError("Agent role is required", 403)
    except AuthError:
        await websocket.close(code=4403)
        return

    await hub.connect_queue(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect_queue(websocket)


async def pre_authorize_chatkit_request(
    body: bytes,
    context: RequestContext,
) -> None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return

    params = payload.get("params", {})
    thread_id = params.get("thread_id")
    if thread_id:
        await store.load_thread(thread_id, context)
