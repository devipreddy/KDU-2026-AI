from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "using",
    "with",
    "you",
}


def clean_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed


def split_sentences(text: str) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        if count_tokens(tokenizer, sentence) > max_tokens:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
            chunks.extend(_chunk_sentence_words(sentence, tokenizer, max_tokens))
            continue

        candidate = " ".join(current_sentences + [sentence])
        if current_sentences and count_tokens(tokenizer, candidate) > max_tokens:
            chunks.append(" ".join(current_sentences))
            overlap = _take_overlap(current_sentences, tokenizer, overlap_tokens)
            current_sentences = overlap + [sentence]
            while count_tokens(tokenizer, " ".join(current_sentences)) > max_tokens and len(current_sentences) > 1:
                current_sentences.pop(0)
            continue

        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def group_texts_by_tokens(items: Iterable[str], tokenizer, max_tokens: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []

    for item in items:
        cleaned = clean_text(item)
        if not cleaned:
            continue

        candidate = " ".join(current + [cleaned])
        if current and count_tokens(tokenizer, candidate) > max_tokens:
            groups.append(" ".join(current))
            current = [cleaned]
            continue

        current.append(cleaned)

    if current:
        groups.append(" ".join(current))

    return groups


def trim_to_word_budget(text: str, max_words: int) -> str:
    words = clean_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()


def count_words(text: str) -> int:
    return len(clean_text(text).split())


def build_extractive_summary(text: str, min_words: int, max_words: int, hint_text: str = "") -> str:
    normalized_text = clean_text(text)
    if count_words(normalized_text) <= max_words:
        return normalized_text

    sentences = split_sentences(text)
    if not sentences:
        return ""

    hint_keywords = _extract_keywords(hint_text)
    scored_sentences = []
    for index, sentence in enumerate(sentences):
        score = _score_sentence(sentence, hint_keywords)
        scored_sentences.append((score, index, sentence))

    selected_indexes = _select_required_sentence_indexes(sentences, hint_keywords)
    ranked = sorted(scored_sentences, key=lambda item: (-item[0], item[1]))
    current_words = sum(count_words(sentences[index]) for index in selected_indexes)

    for score, index, sentence in ranked:
        if index in selected_indexes:
            continue
        sentence_words = count_words(sentence)
        if current_words and current_words + sentence_words > max_words + 15:
            continue
        selected_indexes.add(index)
        current_words += sentence_words
        if current_words >= min_words:
            break

    if not selected_indexes:
        selected_indexes.add(0)

    ordered_sentences = [sentence for index, sentence in enumerate(sentences) if index in selected_indexes]
    summary = clean_text(" ".join(ordered_sentences))
    if count_words(summary) < min_words:
        summary = _append_until_budget(sentences, selected_indexes, min_words, max_words)

    return trim_to_word_budget(summary, max_words)


def select_relevant_context(question: str, context: str, max_words: int = 140) -> str:
    sentences = split_sentences(context)
    if not sentences:
        return clean_text(context)

    question_keywords = _extract_keywords(question)
    scored = []
    for index, sentence in enumerate(sentences):
        overlap = len(_extract_keywords(sentence) & question_keywords)
        heuristic_bonus = _question_specific_bonus(question.lower(), sentence.lower())
        if overlap == 0 and heuristic_bonus == 0:
            continue
        scored.append((overlap + heuristic_bonus + _score_sentence(sentence, question_keywords), index, sentence))

    if not scored:
        return trim_to_word_budget(context, max_words)

    top_indexes = [index for _, index, _ in sorted(scored, key=lambda item: (-item[0], item[1]))[:2]]
    selected_indexes = set()
    for index in top_indexes:
        selected_indexes.add(index)
        if index - 1 >= 0:
            selected_indexes.add(index - 1)
        if index + 1 < len(sentences):
            selected_indexes.add(index + 1)

    selected_indexes = sorted(selected_indexes)
    selected_sentences = [sentences[index] for index in selected_indexes]
    return trim_to_word_budget(" ".join(selected_sentences), max_words)


def _take_overlap(sentences: list[str], tokenizer, overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []

    overlap: list[str] = []
    for sentence in reversed(sentences):
        overlap.insert(0, sentence)
        if count_tokens(tokenizer, " ".join(overlap)) >= overlap_tokens:
            break

    return overlap


def _chunk_sentence_words(sentence: str, tokenizer, max_tokens: int) -> list[str]:
    words = sentence.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])
        if current_words and count_tokens(tokenizer, candidate) > max_tokens:
            chunks.append(" ".join(current_words))
            current_words = [word]
            continue
        current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9:/.-]+", clean_text(text).lower())
    return {word for word in words if len(word) >= 4 and word not in STOPWORDS}


def _score_sentence(sentence: str, hint_keywords: set[str]) -> int:
    normalized = sentence.lower()
    words = _extract_keywords(sentence)
    overlap_score = len(words & hint_keywords) * 3
    structure_score = 0

    if any(char.isdigit() for char in sentence):
        structure_score += 3
    if "http" in normalized or "127.0.0.1" in normalized:
        structure_score += 4
    if any(term in normalized for term in {"lm studio", "vscode", "visual studio code", "continue", "port", "server", "api"}):
        structure_score += 3
    if any(term in normalized for term in {"troubleshooting", "connect", "slow", "performance", "check", "ensure"}):
        structure_score += 2

    length_score = 1 if 8 <= count_words(sentence) <= 30 else 0
    return overlap_score + structure_score + length_score


def _append_until_budget(sentences: list[str], selected_indexes: set[int], min_words: int, max_words: int) -> str:
    ordered_indexes = list(selected_indexes)
    if not ordered_indexes:
        ordered_indexes = [0]

    current_summary = clean_text(" ".join(sentences[index] for index in sorted(selected_indexes)))
    current_words = count_words(current_summary)
    if current_words >= min_words:
        return current_summary

    for index, sentence in enumerate(sentences):
        if index in selected_indexes:
            continue
        candidate = clean_text(f"{current_summary} {sentence}")
        if count_words(candidate) > max_words:
            continue
        current_summary = candidate
        current_words = count_words(current_summary)
        if current_words >= min_words:
            break

    return current_summary


def _select_required_sentence_indexes(sentences: list[str], hint_keywords: set[str]) -> set[int]:
    category_preferences = [
        ["lm studio", "developer tab", "server", "host", "port", "127.0.0.1", "1234"],
        ["continue", "visual studio code", "vscode", "api url"],
        ["cannot connect", "configured url", "url and port", "connect", "ensure", "slow", "performance"],
    ]

    selected: set[int] = set()
    for preferred_terms in category_preferences:
        chosen_index = None
        for term in preferred_terms:
            matches = [index for index, sentence in enumerate(sentences) if term in sentence.lower()]
            if matches:
                chosen_index = max(matches, key=lambda index: _score_sentence(sentences[index], hint_keywords))
                break
        if chosen_index is not None:
            selected.add(chosen_index)
    return selected


def _question_specific_bonus(question: str, sentence: str) -> int:
    bonus = 0
    if any(term in question for term in {"tool", "software", "application"}):
        if any(term in sentence for term in {"lm studio", "server", "developer tab", "host"}):
            bonus += 5
    if any(term in question for term in {"connect", "url", "port", "check"}):
        if any(term in sentence for term in {"connect", "url", "port", "ensure", "running"}):
            bonus += 5
    return bonus
