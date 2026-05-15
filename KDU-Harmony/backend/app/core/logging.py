import logging
import sys

import structlog

from app.core.config import settings
from app.core.observability import redact_structlog_event


def configure_logging(log_level: str | None = None) -> None:
    level = log_level_value(log_level or settings.log_level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            redact_structlog_event,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def log_level_value(log_level: str) -> int:
    normalized = log_level.upper()
    value = getattr(logging, normalized, None)
    if isinstance(value, int):
        return value
    return logging.INFO
