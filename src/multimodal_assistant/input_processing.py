from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from multimodal_assistant.schemas import ImagePayload, NormalizedImage


class InputProcessor:
    def normalize_message(self, message: str, has_image: bool) -> str:
        cleaned = message.strip()
        if cleaned:
            return cleaned
        if has_image:
            return "Describe the uploaded image."
        raise ValueError("A non-empty message is required when no image is provided.")

    def normalize_image(self, image: str | ImagePayload | None) -> NormalizedImage | None:
        if image is None:
            return None

        if isinstance(image, ImagePayload):
            payload = "".join(image.data.split())
            mime_type = image.mime_type
        else:
            payload = "".join(image.split())
            mime_type = None

        mime_type, base64_data = self._extract_data(payload, mime_type)
        raw_bytes = base64.b64decode(base64_data, validate=True)
        if not raw_bytes:
            raise ValueError("Decoded image is empty.")

        inferred_mime = mime_type or self._infer_mime_type(raw_bytes)
        return NormalizedImage(
            base64_data=base64_data,
            mime_type=inferred_mime,
            raw_bytes=raw_bytes,
            size_bytes=len(raw_bytes),
        )

    @staticmethod
    def _extract_data(payload: str, explicit_mime_type: str | None) -> tuple[str | None, str]:
        if payload.startswith("data:"):
            header, _, encoded = payload.partition(",")
            if not encoded:
                raise ValueError("Image data URL is missing a base64 payload.")
            header_parts = header.split(";")
            mime_type = header_parts[0].removeprefix("data:")
            return mime_type, encoded
        return explicit_mime_type, payload

    @staticmethod
    def _infer_mime_type(raw_bytes: bytes) -> str:
        with Image.open(BytesIO(raw_bytes)) as image:
            if image.format and image.get_format_mimetype():
                return image.get_format_mimetype()
            if image.format:
                return f"image/{image.format.lower()}"
        return "image/png"
