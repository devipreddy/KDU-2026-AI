from multimodal_assistant.input_processing import InputProcessor
from multimodal_assistant.schemas import ImagePayload


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+vr8QAAAAASUVORK5CYII="
)


def test_normalize_image_from_data_url() -> None:
    processor = InputProcessor()
    image = processor.normalize_image(f"data:image/png;base64,{TINY_PNG_BASE64}")

    assert image is not None
    assert image.mime_type == "image/png"
    assert image.size_bytes > 0


def test_normalize_image_from_payload_object() -> None:
    processor = InputProcessor()
    image = processor.normalize_image(
        ImagePayload(data=TINY_PNG_BASE64, mime_type="image/png"),
    )

    assert image is not None
    assert image.base64_data == TINY_PNG_BASE64


def test_normalize_message_defaults_for_image_only_request() -> None:
    processor = InputProcessor()

    assert processor.normalize_message("", has_image=True) == "Describe the uploaded image."
