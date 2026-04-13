from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from .agent import TradingAgent
from .evaluation import run_evaluation_suite


class MessageRequest(BaseModel):
    content: str
    user_id: str | None = None
    base_currency: Literal["INR", "USD", "EUR"] = "INR"


class ApprovalRequestBody(BaseModel):
    approved: bool
    reviewer: str = "human-reviewer"
    reason: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = TradingAgent.from_env()
    app.state.agent = agent
    try:
        yield
    finally:
        agent.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="LangGraph Stock Trading Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/threads/{thread_id}/message")
    async def send_message(thread_id: str, payload: MessageRequest, request: Request):
        agent: TradingAgent = request.app.state.agent
        result = agent.run_turn(
            thread_id=thread_id,
            content=payload.content,
            user_id=payload.user_id,
            base_currency=payload.base_currency,
        )
        return jsonable_encoder(result.model_dump(mode="json"))

    @app.post("/threads/{thread_id}/approval")
    async def approve(thread_id: str, payload: ApprovalRequestBody, request: Request):
        agent: TradingAgent = request.app.state.agent
        state = agent.get_state(thread_id)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")
        result = agent.approve(
            thread_id=thread_id,
            approved=payload.approved,
            reviewer=payload.reviewer,
            reason=payload.reason,
        )
        return jsonable_encoder(result.model_dump(mode="json"))

    @app.get("/threads/{thread_id}/state")
    async def get_state(thread_id: str, request: Request):
        agent: TradingAgent = request.app.state.agent
        state = agent.get_state(thread_id)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")
        return jsonable_encoder(state)

    @app.get("/threads/{thread_id}/analytics")
    async def get_analytics(thread_id: str, request: Request):
        agent: TradingAgent = request.app.state.agent
        state = agent.get_state(thread_id)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")
        analytics = agent._build_analytics(state)
        return jsonable_encoder(analytics.model_dump(mode="json") if analytics else {})

    @app.post("/evaluate")
    async def evaluate(request: Request):
        agent: TradingAgent = request.app.state.agent
        report = run_evaluation_suite(agent)
        return jsonable_encoder(report)

    return app


app = create_app()


def run() -> None:
    uvicorn.run("stocks_langgraph.api:app", host="0.0.0.0", port=8000, reload=False)
