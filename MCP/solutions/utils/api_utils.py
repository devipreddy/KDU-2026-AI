"""
Shared API contracts, response envelopes, and domain error helpers.
"""

import json
import uuid

import mcp.types as types

STATUS_SUCCESS = "success"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"

INVALID_PARAMS_CODE = "INVALID_PARAMS"
RESUME_NOT_FOUND = "RESUME_NOT_FOUND"
RESUME_READ_ERROR = "RESUME_READ_ERROR"
JOB_DESCRIPTION_INVALID = "JOB_DESCRIPTION_INVALID"
VECTOR_INDEX_UNAVAILABLE = "VECTOR_INDEX_UNAVAILABLE"
EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
LLM_PROVIDER_TIMEOUT = "LLM_PROVIDER_TIMEOUT"
DOWNSTREAM_CONNECTOR_ERROR = "DOWNSTREAM_CONNECTOR_ERROR"
WORKFLOW_BLOCKED = "WORKFLOW_BLOCKED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
PROMPT_NOT_FOUND = "PROMPT_NOT_FOUND"


class DomainError(Exception):
    """Application-level error that should be surfaced to the model."""

    def __init__(self, code, message, status=STATUS_FAILED, data=None, trace=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.data = data or {}
        self.trace = trace or {}


def ensure_run_id(run_id=None):
    return run_id or str(uuid.uuid4())


def build_envelope(status, data=None, run_id=None, warnings=None, errors=None, trace=None):
    return {
        "status": status,
        "run_id": ensure_run_id(run_id),
        "data": data or {},
        "warnings": warnings or [],
        "errors": errors or [],
        "trace": trace or {},
    }


def _envelope_text(payload, narrative=None):
    if narrative:
        return f"{narrative}\n\n{json.dumps(payload, indent=2)}"
    return json.dumps(payload, indent=2)


def success_tool_result(data, run_id=None, warnings=None, trace=None, narrative=None, status=STATUS_SUCCESS):
    payload = build_envelope(
        status=status,
        data=data,
        run_id=run_id,
        warnings=warnings,
        trace=trace,
    )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=_envelope_text(payload, narrative=narrative))],
        structuredContent=payload,
        isError=False,
    )


def error_tool_result(code, message, run_id=None, data=None, trace=None, status=STATUS_FAILED):
    payload = build_envelope(
        status=status,
        data=data,
        run_id=run_id,
        errors=[{"code": code, "message": message}],
        trace=trace,
    )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2))],
        structuredContent=payload,
        isError=True,
    )


def tool_result_from_exception(exc, run_id=None, trace=None):
    if isinstance(exc, DomainError):
        merged_trace = dict(exc.trace)
        if trace:
            merged_trace.update(trace)
        return error_tool_result(
            code=exc.code,
            message=exc.message,
            run_id=run_id,
            data=exc.data,
            trace=merged_trace,
            status=exc.status,
        )

    return error_tool_result(
        code="UNEXPECTED_ERROR",
        message=str(exc),
        run_id=run_id,
        trace=trace,
    )


def text_resource_result(uri, text, mime_type="application/json"):
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=uri,
                mimeType=mime_type,
                text=text,
            )
        ]
    )


def prompt_result(description, text):
    return types.GetPromptResult(
        description=description,
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=text),
            )
        ],
    )


TOOL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "run_id": {"type": "string"},
        "data": {"type": "object"},
        "warnings": {
            "type": "array",
            "items": {"type": "object"},
        },
        "errors": {
            "type": "array",
            "items": {"type": "object"},
        },
        "trace": {"type": "object"},
    },
    "required": ["status", "run_id", "data", "warnings", "errors", "trace"],
}
