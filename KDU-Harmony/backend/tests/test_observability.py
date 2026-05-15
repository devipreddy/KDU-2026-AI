import json
import os

import structlog
from fastapi import FastAPI

from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.observability import (
    OBSERVABILITY_VERSION,
    REDACTED,
    REDACTED_DOB,
    REDACTED_EMAIL,
    REDACTED_MRN,
    REDACTED_PHONE,
    REDACTED_SSN,
    REDACTED_TOKEN,
    build_langsmith_trace_payload,
    configure_langsmith_environment,
    configure_observability,
    redact_for_observability,
    redact_structlog_event,
)


def test_redact_for_observability_sanitizes_phi_in_nested_payload() -> None:
    payload = {
        "query": (
            "Find MRN 1234567 for jane@example.com, phone 555-123-4567, "
            "DOB 1972-04-08 and SSN 111-22-3333"
        ),
        "patient_ref": "PATIENT_REF_42",
        "direct": {
            "patient_name": "Jane Smith",
            "token": "[PATIENT_REF_42]",
        },
    }

    redacted = redact_for_observability(payload)

    assert redacted["patient_ref"] == REDACTED_TOKEN
    assert redacted["direct"]["patient_name"] == REDACTED
    assert redacted["direct"]["token"] == REDACTED_TOKEN
    assert REDACTED_EMAIL in redacted["query"]
    assert REDACTED_PHONE in redacted["query"]
    assert REDACTED_DOB in redacted["query"]
    assert REDACTED_SSN in redacted["query"]
    assert REDACTED_MRN in redacted["query"]
    assert "jane@example.com" not in redacted["query"]
    assert "555-123-4567" not in redacted["query"]
    assert "1972-04-08" not in redacted["query"]


def test_structlog_redaction_processor_adds_version() -> None:
    event = redact_structlog_event(
        None,
        "info",
        {
            "event": "retrieval_started",
            "query_text": "Patient [PATIENT_REF_42] email jane@example.com",
        },
    )

    assert event["redaction_version"] == OBSERVABILITY_VERSION
    assert event["query_text"] == f"Patient {REDACTED_TOKEN} email {REDACTED_EMAIL}"


def test_configure_logging_emits_phi_redacted_json(capsys) -> None:
    configure_logging("INFO")

    structlog.get_logger("test").info(
        "phi_observability_test",
        patient_ref="PATIENT_REF_42",
        query="MRN 1234567 for jane@example.com",
    )

    captured = capsys.readouterr().out.strip()
    log_record = json.loads(captured)
    assert log_record["event"] == "phi_observability_test"
    assert log_record["patient_ref"] == REDACTED_TOKEN
    assert log_record["query"] == f"{REDACTED_MRN} for {REDACTED_EMAIL}"
    assert log_record["redaction_version"] == OBSERVABILITY_VERSION


def test_langsmith_trace_payload_is_redacted() -> None:
    payload = build_langsmith_trace_payload(
        name="retrieval.phi_aware_search",
        inputs={"query": "DOB 1972-04-08 for [PATIENT_REF_42]"},
        metadata={"patient_ref": "PATIENT_REF_42", "document_id": "DOC-1"},
    )

    assert payload["inputs"]["query"] == f"{REDACTED_DOB} for {REDACTED_TOKEN}"
    assert payload["metadata"]["patient_ref"] == REDACTED_TOKEN
    assert payload["metadata"]["document_id"] == "DOC-1"
    assert payload["metadata"]["redaction_version"] == OBSERVABILITY_VERSION


def test_configure_langsmith_environment_sets_only_when_enabled(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    disabled = Settings(LANGSMITH_TRACING=False)
    assert configure_langsmith_environment(disabled) is False
    assert "LANGSMITH_TRACING" not in os.environ

    enabled = Settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="test-key",
        LANGSMITH_PROJECT="unit-test-project",
    )
    assert configure_langsmith_environment(enabled) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "unit-test-project"
    assert os.environ["LANGCHAIN_PROJECT"] == "unit-test-project"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"


def test_configure_observability_disabled_does_not_require_optional_packages() -> None:
    status = configure_observability(
        FastAPI(),
        Settings(OTEL_ENABLED=False, LANGSMITH_TRACING=False),
    )

    assert status.structured_logging is True
    assert status.phi_redaction is True
    assert status.opentelemetry_enabled is False
    assert status.opentelemetry_configured is False
    assert status.langsmith_enabled is False
    assert status.langsmith_configured is False
