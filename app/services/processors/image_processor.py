from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.services.providers.openai_provider import OpenAIProvider
from app.services.types import ExtractedSection, ExtractionResult


class ImageProcessor:
    def __init__(self, settings: Settings, provider: OpenAIProvider) -> None:
        self.settings = settings
        self.provider = provider

    def extract(self, file_path: Path) -> ExtractionResult:
        image_bytes, mime_type, original_size, processed_size = self._prepare_image(file_path)
        vision = self.provider.extract_image_content(
            image_bytes=image_bytes,
            mime_type=mime_type,
            context_label="uploaded image",
            operation="vision_image_extract",
        )
        text_output = vision.extracted_text.strip()
        section = ExtractedSection(
            text=text_output,
            source="vision_image",
            description=vision.visual_description or None,
            metadata={"layout_notes": vision.layout_notes, "language": vision.detected_language},
        )
        return ExtractionResult(
            sections=[section],
            description=vision.visual_description or None,
            metadata={
                "original_size": original_size,
                "processed_size": processed_size,
                "detected_language": vision.detected_language,
            },
            usage_entries=[vision.usage_entry],
        )

    def _prepare_image(self, file_path: Path) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            original_size = image.size
            image.thumbnail((1600, 1600))
            processed_size = image.size
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
            return buffer.getvalue(), "image/jpeg", original_size, processed_size
