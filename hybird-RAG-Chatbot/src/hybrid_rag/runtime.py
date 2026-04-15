from __future__ import annotations

import os

from .config import get_settings
from .graph import HybridRAGAgent
from .ingestion import KnowledgeBase
from .memory import SessionMemoryManager
from .observability import TraceStore


def build_runtime() -> dict[str, object]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    settings = get_settings()
    kb = KnowledgeBase(settings)
    memory = SessionMemoryManager(settings.session_store_path)
    trace_store = TraceStore(settings.trace_dir)
    agent = HybridRAGAgent(
        settings=settings,
        knowledge_base=kb,
        memory=memory,
        trace_store=trace_store,
    )
    return {
        "settings": settings,
        "kb": kb,
        "memory": memory,
        "trace_store": trace_store,
        "agent": agent,
    }
