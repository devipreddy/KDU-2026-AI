from pathlib import Path

from fastapi.testclient import TestClient

from multimodal_assistant.api.main import create_app
from multimodal_assistant.bootstrap import build_container
from multimodal_assistant.config import Settings


TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+vr8QAAAAASUVORK5CYII="
)


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'assistant.db').as_posix()}",
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        default_location="Boston, US",
        seed_demo_profiles=False,
        openrouter_api_key=None,
        enable_offline_fallback=True,
    )


def test_chat_endpoint_returns_strict_json(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    container = build_container(settings)
    app = create_app(settings=settings, container=container)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "user_id": "image-user",
                "thread_id": "image-thread",
                "message": "Describe this image",
                "mode": "expert",
                "image": {
                    "data": TINY_PNG_BASE64,
                    "mime_type": "image/png",
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert set(payload["data"]) == {
        "response",
        "temperature",
        "summary",
        "location",
        "image_description",
        "objects",
    }
    assert payload["metadata"]["tools_invoked"] == ["describe_uploaded_image"]
    assert payload["metadata"]["thread_id"] == "image-thread"
    assert payload["data"]["image_description"] is not None
