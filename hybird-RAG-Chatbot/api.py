from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.hybrid_rag.runtime import build_runtime


app = FastAPI(title="Hybrid Search RAG Chatbot API", version="1.0.0")
runtime = build_runtime()
kb = runtime["kb"]
memory = runtime["memory"]
agent = runtime["agent"]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    source_type: str
    source_path: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    if not payload.query.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_INPUT", "message": "Query cannot be empty"}},
        )
    return agent.run(session_id=payload.session_id, query=payload.query)


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict:
    try:
        return kb.ingest_api_source(payload.source_type.lower(), payload.source_path.strip())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "SOURCE_NOT_FOUND", "message": str(exc)}},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_INPUT", "message": str(exc)}},
        ) from exc


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_INPUT", "message": "session_id cannot be empty"}},
        )
    return memory.export_session(session_id)
