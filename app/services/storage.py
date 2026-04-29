from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.utils.files import sanitize_filename


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def save_upload(self, file_name: str, sha256: str, content: bytes) -> Path:
        safe_name = sanitize_filename(file_name)
        path = self.settings.upload_dir / f"{sha256}_{safe_name}"
        if not path.exists():
            path.write_bytes(content)
        return path
