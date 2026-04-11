from __future__ import annotations

from io import BytesIO
from typing import Protocol

from langchain.messages import HumanMessage
from PIL import Image

from multimodal_assistant.schemas import ImageAnalysisResult, NormalizedImage


class ImageAnalyzer(Protocol):
    def analyze(self, image: NormalizedImage, prompt: str | None = None) -> ImageAnalysisResult: ...

    def close(self) -> None: ...


class MetadataImageAnalyzer:
    def analyze(self, image: NormalizedImage, prompt: str | None = None) -> ImageAnalysisResult:
        with Image.open(BytesIO(image.raw_bytes)) as uploaded_image:
            width, height = uploaded_image.size
            orientation = "landscape" if width >= height else "portrait"
            format_name = uploaded_image.format or "unknown"
            color_mode = uploaded_image.mode

        prompt_suffix = f" Prompt focus: {prompt.strip()}." if prompt and prompt.strip() else ""
        description = (
            f"Uploaded image detected: {format_name} image, {width}x{height}, "
            f"{orientation}, color mode {color_mode}. "
            "A vision model is not configured, so this analysis is limited to file metadata."
            f"{prompt_suffix}"
        )
        return ImageAnalysisResult(description=description, objects=[])

    def close(self) -> None:
        return None


class LangChainVisionAnalyzer:
    def __init__(self, model) -> None:
        self._structured_model = model.with_structured_output(ImageAnalysisResult)

    def analyze(self, image: NormalizedImage, prompt: str | None = None) -> ImageAnalysisResult:
        task = prompt.strip() if prompt and prompt.strip() else (
            "Describe the uploaded image, summarize the scene, and list visible objects."
        )
        response = self._structured_model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": task},
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image.base64_data,
                            "mime_type": image.mime_type,
                        },
                    ],
                ),
            ],
        )
        if isinstance(response, ImageAnalysisResult):
            return response
        return ImageAnalysisResult.model_validate(response)

    def close(self) -> None:
        return None


class FallbackImageAnalyzer:
    def __init__(self, primary: ImageAnalyzer | None, fallback: ImageAnalyzer) -> None:
        self._primary = primary
        self._fallback = fallback

    def analyze(self, image: NormalizedImage, prompt: str | None = None) -> ImageAnalysisResult:
        if self._primary is not None:
            try:
                return self._primary.analyze(image, prompt)
            except Exception:
                return self._fallback.analyze(image, prompt)
        return self._fallback.analyze(image, prompt)

    def close(self) -> None:
        if self._primary is not None:
            self._primary.close()
        self._fallback.close()
