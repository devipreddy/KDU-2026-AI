from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings

ALLOWED_MIME_TYPES = {"application/pdf", "text/plain"}
PDF_MAGIC = b"%PDF-"
ENCRYPTION_VERSION = "local-xor-v1"


class UploadValidationError(ValueError):
    """Raised when an uploaded document does not meet local validation rules."""


class StorageReadError(ValueError):
    """Raised when a locally encrypted storage object cannot be read."""


@dataclass(frozen=True)
class StoredDocument:
    storage_uri: str
    storage_path: Path
    checksum_sha256: str
    size_bytes: int
    mime_type: str
    encrypted_checksum_sha256: str
    encryption_version: str


def sanitize_filename(filename: str) -> str:
    cleaned = Path(filename or "upload.bin").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("._")
    return cleaned[:180] or "upload.bin"


def validate_upload(*, filename: str, content_type: str | None, content: bytes) -> str:
    if not content:
        raise UploadValidationError("Uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise UploadValidationError("Uploaded file exceeds the configured size limit")

    normalized_content_type = (content_type or "").split(";")[0].lower().strip()
    suffix = Path(filename or "").suffix.lower()

    if normalized_content_type not in ALLOWED_MIME_TYPES:
        if suffix == ".pdf":
            normalized_content_type = "application/pdf"
        elif suffix in {".txt", ".text"}:
            normalized_content_type = "text/plain"
        else:
            raise UploadValidationError("Only PDF and plain text uploads are supported")

    if normalized_content_type == "application/pdf":
        if not content.startswith(PDF_MAGIC):
            raise UploadValidationError("PDF upload failed signature validation")
        return normalized_content_type

    if normalized_content_type == "text/plain":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadValidationError("Text upload must be valid UTF-8") from exc
        return normalized_content_type

    raise UploadValidationError("Only PDF and plain text uploads are supported")


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0
    while sum(len(block) for block in blocks) < length:
        blocks.append(
            hmac.new(
                key,
                nonce + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
        )
        counter += 1
    return b"".join(blocks)[:length]


def encrypt_for_local_storage(content: bytes, *, document_id: uuid.UUID) -> bytes:
    key = hashlib.sha256(settings.document_storage_key.encode()).digest()
    nonce = document_id.bytes
    stream = _keystream(key, nonce, len(content))
    encrypted = bytes(value ^ stream[index] for index, value in enumerate(content))
    header = {
        "version": ENCRYPTION_VERSION,
        "document_id": str(document_id),
        "created_at": datetime.now(UTC).isoformat(),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return len(header_bytes).to_bytes(4, "big") + header_bytes + encrypted


def decrypt_from_local_storage(encrypted_content: bytes, *, document_id: uuid.UUID) -> bytes:
    if len(encrypted_content) < 4:
        raise StorageReadError("Encrypted storage object is malformed")

    header_length = int.from_bytes(encrypted_content[:4], "big")
    header_start = 4
    header_end = header_start + header_length
    if header_length <= 0 or len(encrypted_content) < header_end:
        raise StorageReadError("Encrypted storage object header is malformed")

    try:
        header = json.loads(encrypted_content[header_start:header_end])
    except json.JSONDecodeError as exc:
        raise StorageReadError("Encrypted storage object header is invalid") from exc

    if header.get("version") != ENCRYPTION_VERSION:
        raise StorageReadError("Unsupported local encryption version")
    if header.get("document_id") != str(document_id):
        raise StorageReadError("Encrypted storage object document ID does not match")

    encrypted_payload = encrypted_content[header_end:]
    key = hashlib.sha256(settings.document_storage_key.encode()).digest()
    stream = _keystream(key, document_id.bytes, len(encrypted_payload))
    return bytes(value ^ stream[index] for index, value in enumerate(encrypted_payload))


def read_encrypted_document(*, storage_path: Path, document_id: uuid.UUID) -> bytes:
    try:
        encrypted_content = storage_path.read_bytes()
    except OSError as exc:
        raise StorageReadError("Encrypted storage object could not be read") from exc
    return decrypt_from_local_storage(encrypted_content, document_id=document_id)


def store_encrypted_document(
    *,
    document_id: uuid.UUID,
    original_filename: str,
    content_type: str | None,
    content: bytes,
) -> StoredDocument:
    mime_type = validate_upload(
        filename=original_filename,
        content_type=content_type,
        content=content,
    )
    checksum_sha256 = hashlib.sha256(content).hexdigest()
    encrypted_content = encrypt_for_local_storage(content, document_id=document_id)
    encrypted_checksum_sha256 = hashlib.sha256(encrypted_content).hexdigest()

    storage_root = settings.document_storage_root.resolve()
    relative_dir = Path(datetime.now(UTC).strftime("%Y/%m/%d"))
    storage_dir = storage_root / relative_dir
    storage_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(original_filename)
    storage_path = storage_dir / f"{document_id}_{safe_name}.enc"
    storage_path.write_bytes(encrypted_content)

    return StoredDocument(
        storage_uri=f"local-encrypted://{relative_dir.as_posix()}/{storage_path.name}",
        storage_path=storage_path,
        checksum_sha256=checksum_sha256,
        size_bytes=len(content),
        mime_type=mime_type,
        encrypted_checksum_sha256=encrypted_checksum_sha256,
        encryption_version=ENCRYPTION_VERSION,
    )
