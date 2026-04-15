from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import INJECTION_PATTERNS


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sanitize_context(text: str) -> str:
    cleaned = text
    for pattern in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[filtered]", cleaned, flags=re.IGNORECASE)
    return normalize_whitespace(cleaned)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_json_file(path: str, default: Any) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json_file(path: str, payload: Any) -> None:
    ensure_parent_dir(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def keyword_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine_similarity_from_tokens(left_text: str, right_text: str) -> float:
    left_counter = Counter(keyword_tokenize(left_text))
    right_counter = Counter(keyword_tokenize(right_text))
    if not left_counter or not right_counter:
        return 0.0
    numerator = sum(left_counter[token] * right_counter.get(token, 0) for token in left_counter)
    left_norm = sum(value * value for value in left_counter.values()) ** 0.5
    right_norm = sum(value * value for value in right_counter.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
