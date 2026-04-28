from __future__ import annotations

import re
from collections import Counter

from app.services.types import ExtractedSection, NormalizedTextResult


class TextNormalizer:
    def normalize(self, sections: list[ExtractedSection]) -> NormalizedTextResult:
        extracted_text = "\n\n".join(section.text.strip() for section in sections if section.text.strip())
        repeated_lines = self._find_repeated_lines(sections)

        cleaned_sections: list[ExtractedSection] = []
        for section in sections:
            cleaned_text = self._clean_section_text(section.text, repeated_lines)
            if section.description:
                cleaned_text = f"{cleaned_text}\n\nVisual description:\n{section.description}".strip()
            cleaned_sections.append(
                ExtractedSection(
                    text=cleaned_text,
                    page_number=section.page_number,
                    source=section.source,
                    description=section.description,
                    metadata=section.metadata,
                )
            )

        cleaned_text = "\n\n".join(section.text.strip() for section in cleaned_sections if section.text.strip())
        metadata = {
            "page_count": len({section.page_number for section in cleaned_sections if section.page_number is not None}),
            "repeated_lines_removed": sorted(repeated_lines),
        }
        return NormalizedTextResult(
            sections=cleaned_sections,
            extracted_text=extracted_text.strip(),
            cleaned_text=cleaned_text.strip(),
            metadata=metadata,
        )

    def _find_repeated_lines(self, sections: list[ExtractedSection]) -> set[str]:
        line_counter: Counter[str] = Counter()
        page_counter: Counter[str] = Counter()
        for section in sections:
            lines = [line.strip() for line in section.text.splitlines() if line.strip()]
            seen_on_page: set[str] = set()
            for line in lines:
                normalized = re.sub(r"\s+", " ", line)
                if len(normalized) > 120:
                    continue
                line_counter[normalized] += 1
                seen_on_page.add(normalized)
            for line in seen_on_page:
                page_counter[line] += 1

        repeated: set[str] = set()
        for line, count in page_counter.items():
            if count >= 2 and len(line) >= 3 and not line.isdigit():
                repeated.add(line)
        return repeated

    def _clean_section_text(self, text: str, repeated_lines: set[str]) -> str:
        lines = []
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                lines.append("")
                continue
            if line in repeated_lines:
                continue
            lines.append(line)

        cleaned = "\n".join(lines)
        cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
        return cleaned.strip()
