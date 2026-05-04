from __future__ import annotations

import contextvars
import logging
from datetime import datetime, timezone
from typing import Any

import orjson

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", trace_id_var.get()),
            "session_id": getattr(record, "session_id", session_id_var.get()),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("event_type", "agent", "latency_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return orjson.dumps(payload).decode("utf-8")


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        record.session_id = session_id_var.get()
        return True


_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True
