from types import SimpleNamespace

from app.core.config import settings
from app.services.llm_answer import (
    generate_evidence_grounded_answer,
    response_stream_delta,
    stream_evidence_grounded_answer,
)


class FakeLLMClient:
    provider = "fake_provider"
    model = "fake-model"

    def __init__(self) -> None:
        self.input_text = ""
        self.instructions = ""

    def create_response(self, *, instructions: str, input_text: str):
        self.instructions = instructions
        self.input_text = input_text
        return {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The record shows malaria treatment with artemether [1].",
                        }
                    ]
                }
            ]
        }


class FakeStreamingLLMClient(FakeLLMClient):
    def stream_response(self, *, instructions: str, input_text: str):
        self.instructions = instructions
        self.input_text = input_text
        yield "The record shows "
        yield "malaria treatment [1]."


def rendered_result():
    return SimpleNamespace(
        query="patients with malaria",
        authorization=SimpleNamespace(role_names=["doctor"]),
        rendering_policy=SimpleNamespace(render_mode="full_phi"),
        hits=[
            SimpleNamespace(
                final_rank=1,
                patient_display_ref="PATIENT_REF_42",
                matched_chunk=SimpleNamespace(
                    chunk_id="chunk-1",
                    text="Assessment documents malaria and starts artemether lumefantrine.",
                ),
                parent_context=SimpleNamespace(
                    text="Plan includes repeat malaria smear and infectious disease follow-up."
                ),
                citation=SimpleNamespace(
                    document_id="doc-1",
                    source_document="infectious-disease-note.pdf",
                    section="Assessment",
                    page_number=2,
                    citation_label="DOC-INF-0442 | p. 2 | Assessment",
                    visit_date="2025-06-18",
                    hospital="Mercy West Health",
                    physician="Dr. Samuel Okafor",
                    document_type="clinical_note",
                    diagnosis="Malaria",
                    icd_codes=["B54"],
                ),
                confidence=SimpleNamespace(score=0.91),
            )
        ],
    )


def test_generate_answer_calls_llm_with_authorized_evidence() -> None:
    client = FakeLLMClient()

    answer = generate_evidence_grounded_answer(rendered_result(), client=client)

    assert answer.status == "generated"
    assert answer.provider == "fake_provider"
    assert "malaria treatment" in answer.answer
    assert "Assessment documents malaria" in client.input_text
    assert "Rendered access summary: full_phi" in client.input_text
    assert "Evidence label: DOC-INF-0442 | p. 2 | Assessment" in client.input_text
    assert "Preserve role-aware masking" in client.instructions
    assert "What the records show" in client.instructions
    assert "Why these records matched" in client.instructions
    assert answer.citations[0].citation_label == "DOC-INF-0442 | p. 2 | Assessment"


def test_stream_answer_yields_deltas_and_final_result() -> None:
    client = FakeStreamingLLMClient()

    events = list(stream_evidence_grounded_answer(rendered_result(), client=client))

    assert [event.event for event in events] == ["delta", "delta", "done"]
    assert "".join(event.delta or "" for event in events) == (
        "The record shows malaria treatment [1]."
    )
    assert events[-1].result is not None
    assert events[-1].result.status == "generated"
    assert events[-1].result.answer == "The record shows malaria treatment [1]."
    assert "Assessment documents malaria" in client.input_text


def test_response_stream_delta_parses_responses_api_text_delta() -> None:
    delta = response_stream_delta({"type": "response.output_text.delta", "delta": "partial answer"})

    assert delta == "partial answer"


def test_missing_openai_key_returns_explicit_not_configured_status(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")

    answer = generate_evidence_grounded_answer(rendered_result())

    assert answer.status == "not_configured"
    assert "OPENAI_API_KEY" in answer.answer
    assert answer.citations
