from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import httpx

from app.core.config import settings
from app.core.observability import langsmith_trace, observation_span

LLM_SYNTHESIS_VERSION = "evidence_grounded_llm_answer_v1"
OPENAI_RESPONSES_PROVIDER = "openai_responses_api"
MAX_EVIDENCE_CHARS = 12000
MAX_TEXT_PER_HIT = 1800


class LLMAnswerError(RuntimeError):
    """Raised when the configured LLM provider cannot generate an answer."""


class LLMResponsesClient(Protocol):
    provider: str
    model: str

    def create_response(self, *, instructions: str, input_text: str) -> dict[str, Any]:
        """Create a text response from the configured LLM provider."""


class LLMStreamingResponsesClient(LLMResponsesClient, Protocol):
    def stream_response(self, *, instructions: str, input_text: str) -> Iterator[str]:
        """Yield text deltas from the configured LLM provider."""


@dataclass(frozen=True)
class AnswerCitation:
    index: int
    document_id: str
    chunk_id: str
    citation_label: str
    source_document: str
    section: str | None
    page_number: int | None
    confidence_score: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "citation_label": self.citation_label,
            "source_document": self.source_document,
            "section": self.section,
            "page_number": self.page_number,
            "confidence_score": self.confidence_score,
        }


@dataclass(frozen=True)
class EvidenceItem:
    citation: AnswerCitation
    text: str

    def to_prompt_block(self) -> str:
        citation = self.citation
        source_bits = [
            f"Document ID: {citation.document_id}",
            f"Source: {citation.source_document}",
            f"Section: {citation.section or 'unknown'}",
            f"Page: {citation.page_number if citation.page_number is not None else 'unknown'}",
            f"Confidence: {citation.confidence_score:.3f}",
        ]
        return f"[{citation.index}] {' | '.join(source_bits)}\n{self.text}"


@dataclass(frozen=True)
class LLMAnswerStreamEvent:
    event: str
    delta: str | None = None
    result: LLMAnswerResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class LLMAnswerResult:
    status: str
    answer: str
    provider: str | None
    model: str | None
    citations: list[AnswerCitation]
    latency_ms: float | None = None
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "answerer": LLM_SYNTHESIS_VERSION,
            "status": self.status,
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "citations": [citation.to_metadata() for citation in self.citations],
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class OpenAIResponsesClient:
    provider = OPENAI_RESPONSES_PROVIDER

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_model
        self.timeout_seconds = timeout_seconds or settings.openai_timeout_seconds

    def create_response(self, *, instructions: str, input_text: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": settings.openai_max_output_tokens,
            "store": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def stream_response(self, *, instructions: str, input_text: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": settings.openai_max_output_tokens,
            "store": False,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event_payload = line.removeprefix("data: ").strip()
                    if event_payload == "[DONE]":
                        break
                    try:
                        event = json.loads(event_payload)
                    except json.JSONDecodeError:
                        continue
                    delta = response_stream_delta(event)
                    if delta:
                        yield delta


def generate_evidence_grounded_answer(
    rendered_result: Any,
    *,
    enabled: bool = True,
    client: LLMResponsesClient | None = None,
) -> LLMAnswerResult:
    evidence_items = build_evidence_items(rendered_result)
    citations = [item.citation for item in evidence_items]
    if not enabled:
        return LLMAnswerResult(
            status="disabled",
            answer="LLM answer generation was disabled for this request.",
            provider=None,
            model=None,
            citations=citations,
        )
    if not evidence_items:
        return LLMAnswerResult(
            status="no_evidence",
            answer="No authorized records matched this query.",
            provider=None,
            model=None,
            citations=[],
        )

    active_client = client or configured_openai_client()
    if active_client is None:
        return LLMAnswerResult(
            status="not_configured",
            answer=(
                "LLM answer unavailable: configure OPENAI_API_KEY to generate an "
                "evidence-grounded response. The retrieval evidence is shown below."
            ),
            provider=OPENAI_RESPONSES_PROVIDER,
            model=settings.openai_model,
            citations=citations,
        )

    instructions = answer_instructions(rendered_result)
    input_text = answer_input(rendered_result, evidence_items)
    trace_inputs = {
        "query": getattr(rendered_result, "query", ""),
        "provider": active_client.provider,
        "model": active_client.model,
        "evidence_count": len(evidence_items),
    }
    started_at = perf_counter()
    with (
        observation_span("llm.evidence_answer", attributes=trace_inputs),
        langsmith_trace("llm.evidence_answer", inputs=trace_inputs),
    ):
        try:
            raw_response = active_client.create_response(
                instructions=instructions,
                input_text=input_text,
            )
        except httpx.HTTPStatusError as exc:
            return failed_answer(
                active_client,
                citations,
                started_at=started_at,
                error=f"OpenAI Responses API returned HTTP {exc.response.status_code}",
            )
        except (httpx.RequestError, LLMAnswerError) as exc:
            return failed_answer(
                active_client,
                citations,
                started_at=started_at,
                error=str(exc),
            )

    answer = extract_response_text(raw_response).strip()
    if not answer:
        return failed_answer(
            active_client,
            citations,
            started_at=started_at,
            error="LLM response did not include output text",
        )
    return LLMAnswerResult(
        status="generated",
        answer=answer,
        provider=active_client.provider,
        model=active_client.model,
        citations=citations,
        latency_ms=elapsed_ms(started_at),
    )


def streaming_placeholder_answer(rendered_result: Any, *, enabled: bool = True) -> LLMAnswerResult:
    evidence_items = build_evidence_items(rendered_result)
    citations = [item.citation for item in evidence_items]
    if not enabled:
        return LLMAnswerResult(
            status="disabled",
            answer="LLM answer generation was disabled for this request.",
            provider=None,
            model=None,
            citations=citations,
        )
    if not evidence_items:
        return LLMAnswerResult(
            status="no_evidence",
            answer="No authorized records matched this query.",
            provider=None,
            model=None,
            citations=[],
        )
    return LLMAnswerResult(
        status="streaming",
        answer="",
        provider=OPENAI_RESPONSES_PROVIDER,
        model=settings.openai_model,
        citations=citations,
    )


def stream_evidence_grounded_answer(
    rendered_result: Any,
    *,
    enabled: bool = True,
    client: LLMStreamingResponsesClient | None = None,
) -> Iterator[LLMAnswerStreamEvent]:
    evidence_items = build_evidence_items(rendered_result)
    citations = [item.citation for item in evidence_items]
    if not enabled:
        yield LLMAnswerStreamEvent(
            event="done",
            result=LLMAnswerResult(
                status="disabled",
                answer="LLM answer generation was disabled for this request.",
                provider=None,
                model=None,
                citations=citations,
            ),
        )
        return
    if not evidence_items:
        yield LLMAnswerStreamEvent(
            event="done",
            result=LLMAnswerResult(
                status="no_evidence",
                answer="No authorized records matched this query.",
                provider=None,
                model=None,
                citations=[],
            ),
        )
        return

    active_client = client or configured_openai_client()
    if active_client is None:
        yield LLMAnswerStreamEvent(
            event="done",
            result=LLMAnswerResult(
                status="not_configured",
                answer=(
                    "LLM answer unavailable: configure OPENAI_API_KEY to generate an "
                    "evidence-grounded response. The retrieval evidence is shown below."
                ),
                provider=OPENAI_RESPONSES_PROVIDER,
                model=settings.openai_model,
                citations=citations,
            ),
        )
        return

    instructions = answer_instructions(rendered_result)
    input_text = answer_input(rendered_result, evidence_items)
    trace_inputs = {
        "query": getattr(rendered_result, "query", ""),
        "provider": active_client.provider,
        "model": active_client.model,
        "evidence_count": len(evidence_items),
        "streaming": True,
    }
    started_at = perf_counter()
    parts: list[str] = []
    with (
        observation_span("llm.evidence_answer.stream", attributes=trace_inputs),
        langsmith_trace("llm.evidence_answer.stream", inputs=trace_inputs),
    ):
        try:
            for delta in active_client.stream_response(
                instructions=instructions,
                input_text=input_text,
            ):
                parts.append(delta)
                yield LLMAnswerStreamEvent(event="delta", delta=delta)
        except httpx.HTTPStatusError as exc:
            yield LLMAnswerStreamEvent(
                event="done",
                result=failed_answer(
                    active_client,
                    citations,
                    started_at=started_at,
                    error=f"OpenAI Responses API returned HTTP {exc.response.status_code}",
                ),
            )
            return
        except (httpx.RequestError, LLMAnswerError) as exc:
            yield LLMAnswerStreamEvent(
                event="done",
                result=failed_answer(
                    active_client,
                    citations,
                    started_at=started_at,
                    error=str(exc),
                ),
            )
            return

    answer = "".join(parts).strip()
    if not answer:
        yield LLMAnswerStreamEvent(
            event="done",
            result=failed_answer(
                active_client,
                citations,
                started_at=started_at,
                error="LLM response did not include output text",
            ),
        )
        return
    yield LLMAnswerStreamEvent(
        event="done",
        result=LLMAnswerResult(
            status="generated",
            answer=answer,
            provider=active_client.provider,
            model=active_client.model,
            citations=citations,
            latency_ms=elapsed_ms(started_at),
        ),
    )


def configured_openai_client() -> OpenAIResponsesClient | None:
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        return None
    return OpenAIResponsesClient(api_key=api_key)


def failed_answer(
    client: LLMResponsesClient,
    citations: list[AnswerCitation],
    *,
    started_at: float,
    error: str,
) -> LLMAnswerResult:
    return LLMAnswerResult(
        status="failed",
        answer="LLM answer generation failed. The retrieval evidence is shown below.",
        provider=client.provider,
        model=client.model,
        citations=citations,
        latency_ms=elapsed_ms(started_at),
        error=error,
    )


def answer_instructions(rendered_result: Any) -> str:
    policy = getattr(rendered_result, "rendering_policy", None)
    render_mode = getattr(policy, "render_mode", "unknown")
    return (
        "You are the clinical retrieval explainer for a healthcare semantic search system. "
        "Write a clear, useful, demo-ready answer for the authenticated user, using only the "
        "evidence blocks supplied by the retrieval system. "
        "Use this structure exactly: "
        "1. Start with a short direct answer of one or two sentences. "
        "2. Add a section titled 'What the records show' with two to five concise bullets. "
        "3. Add a section titled 'Why these records matched' explaining the matched diagnosis, "
        "medication, date, document type, or source signal when evidence supports it. "
        "4. Add a section titled 'Caveats' only when evidence is limited, generalized, "
        "de-identified, metadata-only, or potentially incomplete. "
        "Use a professional but readable voice. "
        "Cite every factual sentence or bullet with evidence numbers like [1]. "
        "Do not cite section headings. "
        "Preserve role-aware masking exactly as shown; never infer, restore, or guess hidden PHI. "
        "Do not provide new diagnosis, treatment, or medical advice beyond the records. "
        "If evidence is insufficient, say so plainly and explain what evidence is missing. "
        f"The rendered PHI mode is {render_mode}."
    )


def answer_input(rendered_result: Any, evidence_items: list[EvidenceItem]) -> str:
    role_names = getattr(getattr(rendered_result, "authorization", None), "role_names", [])
    blocks: list[str] = [
        f"Query: {getattr(rendered_result, 'query', '')}",
        f"User roles: {', '.join(role_names) if role_names else 'unknown'}",
        f"Rendered access summary: {rendering_summary(rendered_result)}",
        "Evidence:",
    ]
    used_chars = sum(len(block) for block in blocks)
    for item in evidence_items:
        block = item.to_prompt_block()
        if used_chars + len(block) > MAX_EVIDENCE_CHARS:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def rendering_summary(rendered_result: Any) -> str:
    policy = getattr(rendered_result, "rendering_policy", None)
    render_mode = getattr(policy, "render_mode", "unknown")
    assignment_modes = sorted(
        {
            str(
                (getattr(hit, "retrieval", {}) or {}).get("rendering", {}).get("patient_assignment")
            )
            for hit in getattr(rendered_result, "hits", [])
            if isinstance(getattr(hit, "retrieval", None), dict)
            and isinstance(getattr(hit, "retrieval", {}).get("rendering"), dict)
            and (getattr(hit, "retrieval", {}).get("rendering", {}).get("patient_assignment"))
        }
    )
    if assignment_modes:
        return f"{render_mode}; patient assignment modes: {', '.join(assignment_modes)}"
    return str(render_mode)


def build_evidence_items(rendered_result: Any) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, hit in enumerate(getattr(rendered_result, "hits", []), start=1):
        text = evidence_text_for_hit(hit)
        if not text.strip():
            continue
        citation = hit.citation
        matched_chunk = hit.matched_chunk
        confidence = hit.confidence
        items.append(
            EvidenceItem(
                citation=AnswerCitation(
                    index=index,
                    document_id=str(getattr(citation, "document_id", "")),
                    chunk_id=str(getattr(matched_chunk, "chunk_id", "")),
                    citation_label=str(getattr(citation, "citation_label", "")),
                    source_document=str(getattr(citation, "source_document", "")),
                    section=getattr(citation, "section", None),
                    page_number=getattr(citation, "page_number", None),
                    confidence_score=float(getattr(confidence, "score", 0.0)),
                ),
                text=text[:MAX_TEXT_PER_HIT].strip(),
            )
        )
    return items


def evidence_text_for_hit(hit: Any) -> str:
    citation = hit.citation
    diagnosis = getattr(citation, "diagnosis", None)
    icd_codes = getattr(citation, "icd_codes", [])
    metadata_lines = [
        f"Evidence label: {getattr(citation, 'citation_label', 'unknown')}",
        f"Patient: {getattr(hit, 'patient_display_ref', None) or 'hidden'}",
        f"Visit date: {getattr(citation, 'visit_date', None) or 'unknown'}",
        f"Hospital: {getattr(citation, 'hospital', None) or 'unknown'}",
        f"Physician: {getattr(citation, 'physician', None) or 'unknown'}",
        f"Document type: {getattr(citation, 'document_type', None) or 'unknown'}",
        f"Diagnosis: {diagnosis or 'unknown'}",
        f"ICD codes: {', '.join(icd_codes) if icd_codes else 'unknown'}",
        f"Retrieval confidence: {float(getattr(hit.confidence, 'score', 0.0)):.3f}",
    ]
    text_parts = [
        "Matched chunk:",
        getattr(getattr(hit, "matched_chunk", None), "text", None) or "",
        "Parent context:",
        getattr(getattr(hit, "parent_context", None), "text", None) or "",
    ]
    return "\n".join([*metadata_lines, *text_parts]).strip()


def extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for output_item in response.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") in {"output_text", "text"}:
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def response_stream_delta(event: dict[str, Any]) -> str:
    event_type = str(event.get("type") or "")
    delta = event.get("delta")
    if isinstance(delta, str) and event_type.endswith(".delta"):
        return delta

    text = event.get("text")
    if isinstance(text, str) and event_type.endswith(".delta"):
        return text

    if event_type == "response.output_text.delta" and isinstance(delta, str):
        return delta
    return ""


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)
