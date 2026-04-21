from __future__ import annotations

import logging

import torch

from .config import load_settings
from .models import HuggingFaceQuestionAnswerer, HuggingFaceRefiner, HuggingFaceSummarizer
from .schemas import AnswerResult, AssistantSettings, SummaryArtifacts, SummaryLength
from .text_utils import clean_text

LOGGER = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableLambda
except ImportError:
    RunnableLambda = None


class TriModelAssistant:
    def __init__(
        self,
        settings: AssistantSettings,
        summarizer: HuggingFaceSummarizer,
        refiner: HuggingFaceRefiner,
        question_answerer: HuggingFaceQuestionAnswerer,
    ) -> None:
        self.settings = settings
        self.summarizer = summarizer
        self.refiner = refiner
        self.question_answerer = question_answerer
        self._summary_workflow = self._build_summary_workflow()
        self._qa_workflow = self._build_qa_workflow()

    def generate_summary(self, text: str, length: str | SummaryLength) -> SummaryArtifacts:
        payload = {"text": text, "length": length}
        if self._summary_workflow is not None:
            return self._summary_workflow.invoke(payload)

        prepared = self._prepare_summary_request(payload)
        drafted = self._run_summarization(prepared)
        return self._run_refinement(drafted)

    def answer_question(self, question: str, summary: str) -> AnswerResult:
        payload = {"question": question, "summary": summary}
        if self._qa_workflow is not None:
            return self._qa_workflow.invoke(payload)

        prepared = self._prepare_qa_request(payload)
        return self._run_qa(prepared)

    def _build_summary_workflow(self):
        if RunnableLambda is None:
            LOGGER.warning("LangChain is not installed. Falling back to direct orchestration.")
            return None

        return (
            RunnableLambda(self._prepare_summary_request)
            | RunnableLambda(self._run_summarization)
            | RunnableLambda(self._run_refinement)
        )

    def _build_qa_workflow(self):
        if RunnableLambda is None:
            return None

        return RunnableLambda(self._prepare_qa_request) | RunnableLambda(self._run_qa)

    def _prepare_summary_request(self, payload: dict) -> dict:
        text = clean_text(str(payload.get("text", "")))
        if len(text.split()) < 20:
            raise ValueError("Provide a longer input passage so the summarization stage has enough context.")

        length = SummaryLength.from_value(payload.get("length", SummaryLength.MEDIUM))
        return {"text": text, "length": length}

    def _run_summarization(self, payload: dict) -> dict:
        draft_summary, chunk_count = self.summarizer.summarize(payload["text"])
        return {
            "text": payload["text"],
            "length": payload["length"],
            "draft_summary": draft_summary,
            "chunk_count": chunk_count,
        }

    def _run_refinement(self, payload: dict) -> SummaryArtifacts:
        profile = self.settings.length_profiles[payload["length"]]
        final_summary = self.refiner.refine(payload["text"], payload["draft_summary"], profile)
        return SummaryArtifacts(
            source_text=payload["text"],
            chunk_count=payload["chunk_count"],
            draft_summary=payload["draft_summary"],
            final_summary=final_summary,
            length=payload["length"],
        )

    def _prepare_qa_request(self, payload: dict) -> dict:
        question = clean_text(str(payload.get("question", "")))
        summary = clean_text(str(payload.get("summary", "")))
        if not question:
            raise ValueError("Question cannot be empty.")
        if not summary:
            raise ValueError("Summary cannot be empty.")
        return {"question": question, "summary": summary}

    def _run_qa(self, payload: dict) -> AnswerResult:
        return self.question_answerer.answer(payload["question"], payload["summary"])


def build_assistant(config_path: str | None = None) -> TriModelAssistant:
    settings = load_settings(config_path)
    device = _resolve_device(settings.runtime.device)
    LOGGER.info("Initializing tri-model assistant on device %s", device)

    summarizer = HuggingFaceSummarizer(settings.summarizer, device=device)
    refiner = HuggingFaceRefiner(settings.refiner, device=device)
    question_answerer = HuggingFaceQuestionAnswerer(
        settings.qa,
        device=device,
        confidence_threshold=settings.runtime.qa_confidence_threshold,
    )
    return TriModelAssistant(
        settings=settings,
        summarizer=summarizer,
        refiner=refiner,
        question_answerer=question_answerer,
    )


def _resolve_device(device_setting: str) -> int:
    normalized = device_setting.strip().lower()
    if normalized == "cpu":
        return -1
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested in the config, but no CUDA device is available.")
        return 0
    if normalized == "auto":
        return 0 if torch.cuda.is_available() else -1
    raise ValueError("runtime.device must be one of: auto, cpu, cuda")
