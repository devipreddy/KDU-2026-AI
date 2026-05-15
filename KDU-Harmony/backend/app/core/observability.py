from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import FastAPI

from app.core.config import Settings, settings

OBSERVABILITY_VERSION = "observability_phi_redaction_v1"
REDACTED = "[REDACTED_PHI]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_MRN = "[REDACTED_MRN]"
REDACTED_SSN = "[REDACTED_SSN]"
REDACTED_DOB = "[REDACTED_DOB]"

SENSITIVE_KEY_FRAGMENTS = {
    "address",
    "date_of_birth",
    "dob",
    "email",
    "encrypted_value",
    "first_name",
    "last_name",
    "medical_record_number",
    "mrn",
    "patient_name",
    "phone",
    "secret",
    "ssn",
}

TOKEN_KEY_NAMES = {
    "access_token",
    "patient_ref",
    "patient_id",
    "phi_token",
    "refresh_token",
    "token",
}

PHI_TOKEN_PATTERN = re.compile(
    r"\[(?:PATIENT_REF|DOB|MRN|PHONE|ADDR|ADDRESS|EMAIL|SSN)[A-Z0-9_:-]*\]",
    flags=re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_PATTERN = re.compile(
    r"\b(?:MRN|medical\s+record(?:\s+number)?|record\s+number)\s*[:#-]?\s*"
    r"[A-Z]{0,6}-?\d{4,}(?:-\d+)?\b",
    flags=re.IGNORECASE,
)
DOB_PATTERN = re.compile(
    r"\b(?:DOB|date\s+of\s+birth)\s*[:#-]?\s*"
    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    flags=re.IGNORECASE,
)

_OTEL_CONFIGURED = False


@dataclass(frozen=True)
class ObservabilityStatus:
    structured_logging: bool
    phi_redaction: bool
    opentelemetry_enabled: bool
    opentelemetry_configured: bool
    langsmith_enabled: bool
    langsmith_configured: bool
    warnings: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "observability_version": OBSERVABILITY_VERSION,
            "structured_logging": self.structured_logging,
            "phi_redaction": self.phi_redaction,
            "opentelemetry_enabled": self.opentelemetry_enabled,
            "opentelemetry_configured": self.opentelemetry_configured,
            "langsmith_enabled": self.langsmith_enabled,
            "langsmith_configured": self.langsmith_configured,
            "warnings": self.warnings,
        }


def redact_for_observability(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: redact_for_observability(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_for_observability(item) for item in value]
    if isinstance(value, tuple):
        return [redact_for_observability(item) for item in value]
    if isinstance(value, set):
        return [redact_for_observability(item) for item in value]
    if isinstance(value, str):
        if key_is_sensitive(key):
            return REDACTED
        if key_is_token_like(key):
            return REDACTED_TOKEN
        return redact_phi_in_text(value)
    if key_is_sensitive(key):
        return REDACTED
    if key_is_token_like(key):
        return REDACTED_TOKEN
    return value


def redact_phi_in_text(text: str) -> str:
    redacted = PHI_TOKEN_PATTERN.sub(REDACTED_TOKEN, text)
    redacted = EMAIL_PATTERN.sub(REDACTED_EMAIL, redacted)
    redacted = PHONE_PATTERN.sub(REDACTED_PHONE, redacted)
    redacted = SSN_PATTERN.sub(REDACTED_SSN, redacted)
    redacted = MRN_PATTERN.sub(REDACTED_MRN, redacted)
    return DOB_PATTERN.sub(REDACTED_DOB, redacted)


def key_is_sensitive(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.lower()
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def key_is_token_like(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.lower()
    return normalized in TOKEN_KEY_NAMES or normalized.endswith("_token")


def redact_structlog_event(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    redacted = redact_for_observability(event_dict)
    if isinstance(redacted, dict):
        redacted["redaction_version"] = OBSERVABILITY_VERSION
        return redacted
    return {
        "event": redacted,
        "redaction_version": OBSERVABILITY_VERSION,
    }


def build_langsmith_trace_payload(
    *,
    name: str,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "inputs": redact_for_observability(inputs or {}),
        "metadata": {
            **redact_for_observability(metadata or {}),
            "redaction_version": OBSERVABILITY_VERSION,
        },
    }


def configure_langsmith_environment(active_settings: Settings = settings) -> bool:
    if not active_settings.langsmith_tracing:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = active_settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = active_settings.langsmith_project
    if active_settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = active_settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = active_settings.langsmith_endpoint
    if active_settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = active_settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = active_settings.langsmith_api_key
    return bool(active_settings.langsmith_api_key)


@contextmanager
def langsmith_trace(
    name: str,
    *,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    active_settings: Settings = settings,
) -> Iterator[dict[str, Any]]:
    payload = build_langsmith_trace_payload(name=name, inputs=inputs, metadata=metadata)
    if not active_settings.langsmith_tracing:
        yield payload
        return

    try:
        import langsmith as ls
    except Exception:
        ls = None

    if ls is not None and hasattr(ls, "trace"):
        with ls.trace(
            name,
            "chain",
            project_name=active_settings.langsmith_project,
            inputs=payload["inputs"],
            metadata=payload["metadata"],
        ) as run_tree:
            try:
                yield payload
            except Exception as exc:
                payload["error"] = exc.__class__.__name__
                raise
            finally:
                outputs = payload.get("outputs") or {
                    "status": "error" if "error" in payload else "completed"
                }
                try:
                    run_tree.end(outputs=redact_for_observability(outputs))
                except Exception:
                    pass
        return

    try:
        from langsmith.run_helpers import tracing_context
    except Exception:
        yield payload
        return

    with tracing_context(enabled=True, metadata=payload["metadata"]):
        yield payload


@contextmanager
def observation_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    active_settings: Settings = settings,
) -> Iterator[None]:
    if not active_settings.otel_enabled:
        yield
        return

    try:
        from opentelemetry import trace
    except Exception:
        yield
        return

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        for key, value in flattened_attributes(redact_for_observability(attributes or {})).items():
            span.set_attribute(key, value)
        yield


def configure_observability(
    app: FastAPI, active_settings: Settings = settings
) -> ObservabilityStatus:
    warnings: list[str] = []
    langsmith_configured = configure_langsmith_environment(active_settings)
    opentelemetry_configured = configure_opentelemetry_app(
        app,
        active_settings=active_settings,
        warnings=warnings,
    )
    return ObservabilityStatus(
        structured_logging=True,
        phi_redaction=True,
        opentelemetry_enabled=active_settings.otel_enabled,
        opentelemetry_configured=opentelemetry_configured,
        langsmith_enabled=active_settings.langsmith_tracing,
        langsmith_configured=langsmith_configured,
        warnings=warnings,
    )


def configure_opentelemetry_app(
    app: FastAPI,
    *,
    active_settings: Settings,
    warnings: list[str],
) -> bool:
    global _OTEL_CONFIGURED

    if not active_settings.otel_enabled:
        return False
    if _OTEL_CONFIGURED:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:
        warnings.append(f"opentelemetry_unavailable:{exc.__class__.__name__}")
        return False

    resource = Resource.create(
        {
            "service.name": active_settings.otel_service_name or active_settings.project_name,
            "deployment.environment": active_settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)
    if active_settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(
            endpoint=active_settings.otel_exporter_otlp_endpoint,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls=active_settings.otel_excluded_urls,
    )
    if active_settings.otel_sqlalchemy_instrumentation:
        try:
            from app.db.session import engine

            SQLAlchemyInstrumentor().instrument(engine=engine)
        except Exception as exc:
            warnings.append(f"sqlalchemy_instrumentation_failed:{exc.__class__.__name__}")

    _OTEL_CONFIGURED = True
    return True


def flattened_attributes(
    value: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        attribute_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flattened_attributes(item, prefix=attribute_key))
            continue
        if isinstance(item, str | int | float | bool):
            flattened[attribute_key] = item
            continue
        if item is None:
            continue
        flattened[attribute_key] = str(item)
    return flattened


def log_observability_status(status: ObservabilityStatus) -> None:
    logger = structlog.get_logger(__name__)
    logger.info("observability_configured", **status.to_metadata())
