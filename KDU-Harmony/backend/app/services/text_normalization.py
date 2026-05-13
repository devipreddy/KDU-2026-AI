from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_ALIASES = {
    "assessment": "Assessment",
    "chief complaint": "Chief Complaint",
    "diagnosis": "Diagnosis",
    "diagnoses": "Diagnosis",
    "discharge summary": "Discharge Summary",
    "history": "History",
    "history of present illness": "History Of Present Illness",
    "icd": "ICD-10",
    "icd 10": "ICD-10",
    "icd-10": "ICD-10",
    "lab results": "Lab Results",
    "laboratory results": "Lab Results",
    "medication": "Medications",
    "medications": "Medications",
    "plan": "Plan",
    "prescription": "Prescription",
    "treatment plan": "Treatment Plan",
}

MEDICAL_CORRECTIONS = {
    "Arrhythrnia": "Arrhythmia",
    "Atrial fibrilatlon": "Atrial fibrillation",
    "Bcta blockers": "Beta blockers",
    "Beta blockcrs": "Beta blockers",
    "Cardiac arrythmia": "Cardiac arrhythmia",
    "Congestlve heart failure": "Congestive heart failure",
    "Hvpcrtension": "Hypertension",
    "Hyper tension": "Hypertension",
    "Metforrnin": "Metformin",
    "Myocardia1": "Myocardial",
    "SOB": "Shortness of breath",
    "Shortncss of breath": "Shortness of breath",
}

OCR_ARTIFACT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
    "\ufffd": "",
    "¬": "",
    "|": " ",
}


@dataclass(frozen=True)
class NormalizationStats:
    original_char_count: int
    normalized_char_count: int
    whitespace_normalized: bool
    heading_count: int
    paragraph_count: int
    artifact_replacements: int
    medical_corrections: dict[str, int]


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    stats: NormalizationStats


def normalize_medical_text(raw_text: str) -> NormalizationResult:
    text, artifact_count = replace_ocr_artifacts(raw_text)
    text = normalize_line_endings(text)
    text = repair_hyphenated_line_breaks(text)
    text = remove_control_characters(text)
    text, correction_counts = apply_medical_corrections(text)
    lines, heading_count = normalize_lines_and_headings(text)
    normalized_text = restore_paragraphs(lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text).strip()

    return NormalizationResult(
        text=normalized_text,
        stats=NormalizationStats(
            original_char_count=len(raw_text),
            normalized_char_count=len(normalized_text),
            whitespace_normalized=raw_text != normalized_text,
            heading_count=heading_count,
            paragraph_count=count_paragraphs(normalized_text),
            artifact_replacements=artifact_count,
            medical_corrections=correction_counts,
        ),
    )


def replace_ocr_artifacts(text: str) -> tuple[str, int]:
    artifact_count = 0
    cleaned = text
    for artifact, replacement in OCR_ARTIFACT_REPLACEMENTS.items():
        occurrences = cleaned.count(artifact)
        artifact_count += occurrences
        cleaned = cleaned.replace(artifact, replacement)

    cleaned, repeated_punctuation = re.subn(r"([:;,.]){2,}", r"\1", cleaned)
    artifact_count += repeated_punctuation
    cleaned, underline_count = re.subn(r"_{2,}", " ", cleaned)
    artifact_count += underline_count
    return cleaned, artifact_count


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def repair_hyphenated_line_breaks(text: str) -> str:
    return re.sub(r"(?<=\w)-\n\s*(?=\w)", "", text)


def remove_control_characters(text: str) -> str:
    return "".join(character for character in text if character == "\n" or character >= " ")


def apply_medical_corrections(text: str) -> tuple[str, dict[str, int]]:
    corrected = text
    counts: dict[str, int] = {}
    for misspelling, replacement in MEDICAL_CORRECTIONS.items():
        pattern = re.compile(rf"\b{re.escape(misspelling)}\b", flags=re.IGNORECASE)
        corrected, count = pattern.subn(replacement, corrected)
        if count:
            counts[replacement] = counts.get(replacement, 0) + count
    return corrected, counts


def normalize_lines_and_headings(text: str) -> tuple[list[str], int]:
    normalized_lines: list[str] = []
    heading_count = 0

    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            normalized_lines.append("")
            continue

        heading = canonical_heading_line(line)
        if heading:
            normalized_lines.append(heading)
            heading_count += 1
            continue

        normalized_lines.append(line)

    return normalized_lines, heading_count


def canonical_heading_line(line: str) -> str | None:
    stripped = line.strip(" :-")
    canonical = HEADING_ALIASES.get(normalize_heading_key(stripped))
    if canonical:
        return f"{canonical}:"

    match = re.match(r"^([A-Za-z][A-Za-z0-9 /-]{1,40})\s*[:\-]\s*(.*)$", line)
    if not match:
        return None

    heading_key = normalize_heading_key(match.group(1))
    canonical = HEADING_ALIASES.get(heading_key)
    if not canonical:
        return None

    value = match.group(2).strip()
    if value:
        return f"{canonical}: {value}"
    return f"{canonical}:"


def normalize_heading_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace(".", "")).lower().strip()


def restore_paragraphs(lines: list[str]) -> str:
    blocks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            blocks.append(" ".join(current).strip())
            current.clear()

    for line in lines:
        if not line:
            flush_current()
            continue

        if is_heading_line(line):
            flush_current()
            blocks.append(line)
            continue

        current.append(line)

    flush_current()
    return "\n\n".join(block for block in blocks if block)


def is_heading_line(line: str) -> bool:
    if ":" not in line:
        return False
    heading = line.split(":", 1)[0]
    return normalize_heading_key(heading) in {
        normalize_heading_key(value) for value in HEADING_ALIASES.values()
    }


def count_paragraphs(text: str) -> int:
    return len([block for block in text.split("\n\n") if block.strip()])
