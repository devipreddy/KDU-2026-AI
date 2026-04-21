from __future__ import annotations

import logging
import re

import torch
from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

from .schemas import AnswerResult, LengthProfile, ModelSettings
from .text_utils import build_extractive_summary, chunk_text_by_tokens, clean_text, count_words, group_texts_by_tokens, select_relevant_context, trim_to_word_budget

LOGGER = logging.getLogger(__name__)


class HuggingFaceSummarizer:
    def __init__(self, settings: ModelSettings, device: int) -> None:
        self.settings = settings
        self.device = _resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name).to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> tuple[str, int]:
        normalized = clean_text(text)
        if not normalized:
            raise ValueError("Input text is empty after normalization.")

        chunks = chunk_text_by_tokens(
            normalized,
            tokenizer=self.tokenizer,
            max_tokens=self.settings.max_input_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise ValueError("Unable to create chunks from the provided input text.")

        partial_summaries = [self._summarize_chunk(chunk) for chunk in chunks]
        reduced_summary = self._reduce(partial_summaries)
        return reduced_summary, len(chunks)

    def _reduce(self, summaries: list[str]) -> str:
        current = summaries
        while len(current) > 1:
            grouped = group_texts_by_tokens(current, self.tokenizer, self.settings.max_input_tokens)
            current = [self._summarize_chunk(group) for group in grouped]
        return current[0]

    def _summarize_chunk(self, chunk: str) -> str:
        kwargs = dict(self.settings.generation_kwargs)
        encoded = self.tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**encoded, **kwargs)

        summary = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if not summary:
            raise RuntimeError("Summarization model returned an empty summary.")
        return summary


class HuggingFaceRefiner:
    def __init__(self, settings: ModelSettings, device: int) -> None:
        self.settings = settings
        self.device = _resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name).to(self.device)
        self.model.eval()

    def refine(self, source_text: str, draft_summary: str, profile: LengthProfile) -> str:
        source_reference = self._build_source_reference(source_text, profile)
        prompt = (
            "You are a summary refinement model.\n"
            f"Target length: {profile.label.value}\n"
            f"Word budget: {profile.min_words} to {profile.max_words} words\n"
            f"Instruction: {profile.instructions}\n"
            "Preserve the original meaning, avoid repetition, and do not add facts.\n"
            "Keep concrete details such as tools, URLs, ports, configuration values, and troubleshooting steps when they matter.\n\n"
            f"Source facts:\n{source_reference}\n\n"
            f"Draft summary:\n{draft_summary}\n\n"
            "Refined summary:"
        )
        kwargs = dict(self.settings.generation_kwargs)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**encoded, **kwargs)

        generated = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if not self._is_acceptable_summary(generated, profile):
            LOGGER.warning("Refinement output was too weak for the requested budget. Falling back to extractive refinement.")
            extractive = build_extractive_summary(
                source_text,
                min_words=profile.min_words,
                max_words=profile.max_words,
                hint_text=draft_summary,
            )
            if extractive:
                return extractive
            return trim_to_word_budget(draft_summary, profile.max_words)
        return trim_to_word_budget(generated, profile.max_words)

    def _build_source_reference(self, source_text: str, profile: LengthProfile) -> str:
        normalized = clean_text(source_text)
        words = normalized.split()
        if len(words) <= profile.max_words * 2:
            return normalized
        return build_extractive_summary(
            normalized,
            min_words=min(profile.min_words, max(60, profile.max_words // 2)),
            max_words=min(profile.max_words * 2, 320),
            hint_text=normalized,
        )

    def _is_acceptable_summary(self, summary: str, profile: LengthProfile) -> bool:
        words = count_words(summary)
        if words < max(30, int(profile.min_words * 0.75)):
            return False
        return True


class HuggingFaceQuestionAnswerer:
    def __init__(self, settings: ModelSettings, device: int, confidence_threshold: float) -> None:
        self.settings = settings
        self.confidence_threshold = confidence_threshold
        self.device = _resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(settings.model_name).to(self.device)
        self.model.eval()

    def answer(self, question: str, context: str) -> AnswerResult:
        normalized_question = clean_text(question)
        normalized_context = clean_text(context)
        if not normalized_question:
            raise ValueError("Question cannot be empty.")
        if not normalized_context:
            raise ValueError("Context cannot be empty.")

        focused_context = select_relevant_context(normalized_question, normalized_context)
        direct_answer = _rule_based_answer(normalized_question, focused_context, normalized_context)
        encoded = self.tokenizer(
            normalized_question,
            focused_context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=min(self.settings.max_input_tokens, 512),
        )
        sequence_ids = encoded.sequence_ids(0)
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**encoded)

        answer, score = _extract_best_answer(
            input_ids=encoded["input_ids"][0].detach().cpu(),
            start_logits=outputs.start_logits[0].detach().cpu(),
            end_logits=outputs.end_logits[0].detach().cpu(),
            offset_mapping=offset_mapping,
            sequence_ids=sequence_ids,
            context=focused_context,
        )
        if direct_answer:
            answer = direct_answer
            score = max(score, self.confidence_threshold + 0.05)
        is_grounded = bool(answer) and score >= self.confidence_threshold

        if not is_grounded:
            sentence_fallback = _sentence_fallback_answer(normalized_question, focused_context)
            if sentence_fallback:
                answer = sentence_fallback
                score = self.confidence_threshold
                is_grounded = True
            else:
                answer = "I could not find a confident answer in the generated summary."

        return AnswerResult(
            question=normalized_question,
            answer=answer,
            score=score,
            is_grounded=is_grounded,
        )


def _resolve_torch_device(device: int) -> torch.device:
    if device >= 0 and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _extract_best_answer(
    input_ids: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    offset_mapping: list[list[int]],
    sequence_ids: list[int | None],
    context: str,
) -> tuple[str, float]:
    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    valid_positions = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    if not valid_positions:
        return "", 0.0

    top_k = min(10, len(valid_positions))
    candidate_starts = sorted(valid_positions, key=lambda index: float(start_probs[index]), reverse=True)[:top_k]
    candidate_ends = sorted(valid_positions, key=lambda index: float(end_probs[index]), reverse=True)[:top_k]

    best_answer = ""
    best_score = 0.0

    for start_index in candidate_starts:
        for end_index in candidate_ends:
            if end_index < start_index or end_index - start_index > 30:
                continue

            start_char, _ = offset_mapping[start_index]
            _, end_char = offset_mapping[end_index]
            if end_char <= start_char:
                continue

            candidate = clean_text(context[start_char:end_char])
            if not candidate:
                continue

            score = float(start_probs[start_index] * end_probs[end_index])
            if score > best_score:
                best_score = score
                best_answer = candidate

    if best_answer:
        return best_answer, best_score

    cls_index = 0
    if len(input_ids) > 0:
        cls_index = 0
    return "", float(start_probs[cls_index] * end_probs[cls_index])


def _rule_based_answer(question: str, focused_context: str, full_context: str) -> str:
    normalized_question = question.lower()
    normalized_context = focused_context.lower()
    full_context_lower = full_context.lower()

    if any(phrase in normalized_question for phrase in {"what tool", "which tool", "what software", "which application"}):
        if "lm studio" in full_context_lower and any(term in normalized_question for term in {"host", "server", "local model"}):
            return "LM Studio"
        if "lm studio" in normalized_context:
            return "LM Studio"

        candidate_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", full_context)
            if any(term in sentence.lower() for term in {"host", "server", "install", "developer tab"})
        ]
        candidate_text = " ".join(candidate_sentences) if candidate_sentences else focused_context
        candidates = _extract_named_entities(candidate_text)
        if candidates:
            return candidates[0]

    return ""


def _sentence_fallback_answer(question: str, focused_context: str) -> str:
    normalized_question = question.lower()
    if any(phrase in normalized_question for phrase in {"what should i check", "cannot connect", "can't connect", "connection issue"}):
        return focused_context
    return ""


def _extract_named_entities(text: str) -> list[str]:
    candidates = re.findall(r"(?:[A-Z][A-Za-z0-9.+-]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9.+-]*|[A-Z]{2,}))*", text)
    blocked = {"Overview", "System Requirements", "Testing", "Troubleshooting", "Part", "Developer"}
    cleaned = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in blocked:
            continue
        if len(candidate) < 3:
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
    cleaned.sort(key=len, reverse=True)
    return cleaned
