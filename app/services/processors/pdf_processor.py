from __future__ import annotations

from pathlib import Path

import fitz
import pdfplumber

from app.core.config import Settings
from app.services.providers.openai_provider import OpenAIProvider
from app.services.types import ExtractedSection, ExtractionResult


class PDFProcessor:
    def __init__(self, settings: Settings, provider: OpenAIProvider) -> None:
        self.settings = settings
        self.provider = provider

    def extract(self, file_path: Path) -> ExtractionResult:
        sections: list[ExtractedSection] = []
        usage_entries = []
        page_details = []

        with fitz.open(file_path) as document, pdfplumber.open(file_path) as plumber_document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                plumber_page = plumber_document.pages[page_index]
                local_text = self._extract_local_text(page, plumber_page)
                image_count = len(page.get_images(full=True))

                strategy = "local_text"
                description = None
                layout_notes = None
                text_output = local_text

                if self._should_use_vision(local_text, image_count):
                    strategy = "vision_ocr"
                    page_bytes = self._render_page_to_png(page)
                    vision = self.provider.extract_image_content(
                        image_bytes=page_bytes,
                        mime_type="image/png",
                        context_label=f"PDF page {page_number}",
                        operation="vision_pdf_page",
                    )
                    text_output = vision.extracted_text or local_text
                    description = vision.visual_description or None
                    layout_notes = vision.layout_notes or None
                    usage_entries.append(vision.usage_entry)
                elif image_count > 0:
                    page_bytes = self._render_page_to_png(page)
                    visual_description = self.provider.describe_visuals(
                        image_bytes=page_bytes,
                        mime_type="image/png",
                        context_label=f"PDF page {page_number}",
                        operation="vision_pdf_visuals",
                        extracted_text_hint=local_text[:2000] if local_text else None,
                    )
                    if visual_description.contains_meaningful_visuals and visual_description.visual_description:
                        description = visual_description.visual_description
                    layout_notes = visual_description.layout_notes or None
                    usage_entries.append(visual_description.usage_entry)

                sections.append(
                    ExtractedSection(
                        text=text_output.strip(),
                        page_number=page_number,
                        source=strategy,
                        description=description,
                        metadata={"image_count": image_count, "layout_notes": layout_notes},
                    )
                )
                page_details.append(
                    {
                        "page_number": page_number,
                        "strategy": strategy,
                        "image_count": image_count,
                        "character_count": len(text_output.strip()),
                        "has_visual_description": bool(description),
                        "layout_notes": layout_notes,
                    }
                )

        return ExtractionResult(
            sections=sections,
            metadata={"page_details": page_details, "page_count": len(page_details)},
            usage_entries=usage_entries,
        )

    def _extract_local_text(self, page: fitz.Page, plumber_page: pdfplumber.page.Page) -> str:
        fitz_text = page.get_text("text") or ""
        plumber_text = plumber_page.extract_text(layout=True) or ""
        candidates = [fitz_text.strip(), plumber_text.strip()]
        return max(candidates, key=len, default="").strip()

    def _should_use_vision(self, local_text: str, image_count: int) -> bool:
        char_count = len(local_text.strip())
        if char_count < self.settings.pdf_text_threshold_chars:
            return True
        if image_count >= self.settings.pdf_image_trigger_count and char_count < self.settings.pdf_text_threshold_chars * 6:
            return True
        return False

    def _render_page_to_png(self, page: fitz.Page) -> bytes:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return pixmap.tobytes("png")
