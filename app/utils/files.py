from __future__ import annotations

import hashlib
import mimetypes
import re
from pathlib import Path


SUPPORTED_FILE_TYPES: dict[str, tuple[str, ...]] = {
    "pdf": (".pdf",),
    "image": (".jpg", ".jpeg", ".png"),
    "audio": (".mp3", ".wav"),
}


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sanitize_filename(file_name: str) -> str:
    clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_name).strip("._")
    return clean_name or "upload"


def detect_file_type(file_name: str, content_type: str | None = None) -> tuple[str, str]:
    suffix = Path(file_name).suffix.lower()
    guessed_mime = content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"

    for file_type, suffixes in SUPPORTED_FILE_TYPES.items():
        if suffix in suffixes:
            return file_type, guessed_mime

    raise ValueError(f"Unsupported file type for {file_name!r}.")
