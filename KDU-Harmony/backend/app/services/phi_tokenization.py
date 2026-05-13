from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.phi_mapping import PhiMapping

PHI_ENCRYPTION_VERSION = "local-phi-token-v1"

NEXT_LABEL_LOOKAHEAD = (
    r"(?=\s+(?:patient\s+name|patient|name|dob|date\s+of\s+birth|birth\s+date|mrn|"
    r"medical\s+record|record\s+number|phone|telephone|email|e-mail|address|"
    r"diagnosis|assessment|plan|medications?)\b|[.;,\n]|$)"
)


@dataclass(frozen=True)
class PhiPattern:
    entity_type: str
    pattern: re.Pattern[str]
    confidence: float


@dataclass(frozen=True)
class PhiDetection:
    entity_type: str
    value: str
    token: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class PhiTokenizationResult:
    text: str
    detections: list[PhiDetection]

    @property
    def entity_counts(self) -> dict[str, int]:
        return dict(Counter(detection.entity_type for detection in self.detections))

    @property
    def token_count(self) -> int:
        return len({detection.token for detection in self.detections})

    def metadata_summary(self) -> dict:
        token_summaries = sorted(
            {(detection.token, detection.entity_type) for detection in self.detections}
        )
        return {
            "detector": "rule_based_phi_v1",
            "detected": bool(self.detections),
            "entity_counts": self.entity_counts,
            "token_count": self.token_count,
            "tokens": [
                {"token": token, "entity_type": entity_type}
                for token, entity_type in token_summaries
            ],
        }


PHI_PATTERNS = (
    PhiPattern(
        entity_type="PATIENT_NAME",
        pattern=re.compile(
            rf"\b(?:patient\s+name|patient|name)\s*[:#-]\s*"
            rf"(?P<value>[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){{1,3}}?)"
            rf"{NEXT_LABEL_LOOKAHEAD}",
            re.IGNORECASE,
        ),
        confidence=0.92,
    ),
    PhiPattern(
        entity_type="DOB",
        pattern=re.compile(
            rf"\b(?:dob|date\s+of\s+birth|birth\s+date)\s*[:#-]\s*"
            rf"(?P<value>(?:\d{{4}}-\d{{1,2}}-\d{{1,2}})|"
            rf"(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})|"
            rf"(?:[A-Z][a-z]+\s+\d{{1,2}},\s+\d{{4}}))"
            rf"{NEXT_LABEL_LOOKAHEAD}",
            re.IGNORECASE,
        ),
        confidence=0.95,
    ),
    PhiPattern(
        entity_type="MRN",
        pattern=re.compile(
            r"\b(?:mrn|medical\s+record(?:\s+number)?|record\s+number)"
            r"\s*[:#-]?\s*(?P<value>[A-Z]{0,6}-?\d{4,}(?:-\d+)?)\b",
            re.IGNORECASE,
        ),
        confidence=0.96,
    ),
    PhiPattern(
        entity_type="EMAIL",
        pattern=re.compile(
            r"\b(?P<value>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
            re.IGNORECASE,
        ),
        confidence=0.98,
    ),
    PhiPattern(
        entity_type="PHONE",
        pattern=re.compile(
            r"(?<!\d)(?P<value>(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})"
            r"[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)"
        ),
        confidence=0.95,
    ),
    PhiPattern(
        entity_type="ADDRESS",
        pattern=re.compile(
            r"\b(?P<value>\d{1,6}\s+[A-Za-z0-9.' -]+?"
            r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
            r"Drive|Dr\.?|Lane|Ln\.?|Way|Court|Ct\.?)"
            r"(?:,\s*[A-Za-z .'-]+)?(?:,\s*[A-Z]{2})?(?:\s+\d{5}(?:-\d{4})?)?)\b",
            re.IGNORECASE,
        ),
        confidence=0.88,
    ),
)


def tokenize_phi_text(text: str, *, patient_ref: str) -> PhiTokenizationResult:
    detections = detect_phi(text, patient_ref=patient_ref)
    tokenized_text = replace_phi_with_tokens(text, detections)
    return PhiTokenizationResult(text=tokenized_text, detections=detections)


def detect_phi(text: str, *, patient_ref: str) -> list[PhiDetection]:
    candidates: list[PhiDetection] = []
    for phi_pattern in PHI_PATTERNS:
        for match in phi_pattern.pattern.finditer(text):
            value_start, value_end = match.span("value")
            value, start, end = trim_span(text[value_start:value_end], value_start, value_end)
            if not value:
                continue
            candidates.append(
                PhiDetection(
                    entity_type=phi_pattern.entity_type,
                    value=value,
                    token=token_for_phi(
                        entity_type=phi_pattern.entity_type,
                        value=value,
                        patient_ref=patient_ref,
                    ),
                    start=start,
                    end=end,
                    confidence=phi_pattern.confidence,
                )
            )

    return dedupe_overlapping_detections(candidates)


def replace_phi_with_tokens(text: str, detections: list[PhiDetection]) -> str:
    tokenized = text
    for detection in sorted(detections, key=lambda item: item.start, reverse=True):
        tokenized = tokenized[: detection.start] + detection.token + tokenized[detection.end :]
    return tokenized


def trim_span(value: str, start: int, end: int) -> tuple[str, int, int]:
    leading_trimmed = len(value) - len(value.lstrip())
    trailing_trimmed = len(value.rstrip())
    trimmed_value = value.strip()
    return trimmed_value, start + leading_trimmed, start + trailing_trimmed


def dedupe_overlapping_detections(candidates: list[PhiDetection]) -> list[PhiDetection]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.start, -(item.end - item.start), -item.confidence),
    )
    accepted: list[PhiDetection] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for candidate in ordered:
        span_key = (candidate.start, candidate.end, candidate.entity_type)
        if span_key in seen_spans:
            continue
        if any(
            candidate.start < accepted_item.end and accepted_item.start < candidate.end
            for accepted_item in accepted
        ):
            continue
        accepted.append(candidate)
        seen_spans.add(span_key)

    return sorted(accepted, key=lambda item: item.start)


def token_for_phi(*, entity_type: str, value: str, patient_ref: str) -> str:
    patient_token = sanitize_token_part(patient_ref)
    if entity_type == "PATIENT_NAME":
        return f"[{patient_token}]"

    prefix = {
        "DOB": "DOB",
        "MRN": "MRN",
        "PHONE": "PHONE",
        "ADDRESS": "ADDR",
        "EMAIL": "EMAIL",
    }.get(entity_type, entity_type)
    suffix = patient_token.removeprefix("PATIENT_REF_") or patient_token
    digest = (
        hmac.new(
            settings.document_storage_key.encode(),
            f"{patient_token}:{entity_type}:{canonicalize_phi_value(entity_type, value)}".encode(),
            hashlib.sha256,
        )
        .hexdigest()[:8]
        .upper()
    )
    return f"[{prefix}_{suffix}_{digest}]"


def sanitize_token_part(value: str) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", value.upper()).strip("_") or "UNKNOWN"


def canonicalize_phi_value(entity_type: str, value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if entity_type == "EMAIL":
        return normalized.lower()
    if entity_type in {"MRN", "PHONE"}:
        return re.sub(r"[^A-Za-z0-9]+", "", normalized).upper()
    return normalized.lower()


def tokenize_phi_for_document(
    db: Session,
    *,
    document: Document,
    text: str,
) -> PhiTokenizationResult:
    result = tokenize_phi_text(text, patient_ref=document.patient_ref)
    upsert_phi_mappings(
        db,
        detections=result.detections,
        patient_ref=document.patient_ref,
        created_by_user_id=document.created_by_user_id,
    )
    return result


def upsert_phi_mappings(
    db: Session,
    *,
    detections: list[PhiDetection],
    patient_ref: str,
    created_by_user_id: uuid.UUID | None,
) -> list[PhiMapping]:
    first_detection_by_token: dict[str, PhiDetection] = {}
    for detection in detections:
        first_detection_by_token.setdefault(detection.token, detection)

    tokens = list(first_detection_by_token)
    if not tokens:
        return []

    existing_mappings = db.scalars(select(PhiMapping).where(PhiMapping.token.in_(tokens))).all()
    mappings_by_token = {mapping.token: mapping for mapping in existing_mappings}

    for token, detection in first_detection_by_token.items():
        if token in mappings_by_token:
            continue
        mapping = PhiMapping(
            patient_ref=patient_ref,
            token=token,
            entity_type=detection.entity_type,
            encrypted_value=encrypt_phi_value(detection.value, token=token),
            encryption_key_id=settings.encryption_key_id,
            created_by_user_id=created_by_user_id,
        )
        db.add(mapping)
        mappings_by_token[token] = mapping

    db.flush()
    return [mappings_by_token[token] for token in tokens]


def encrypt_phi_value(value: str, *, token: str) -> str:
    value_bytes = value.encode("utf-8")
    nonce = nonce_for_token(token)
    stream = phi_keystream(nonce, len(value_bytes))
    encrypted_payload = bytes(byte ^ stream[index] for index, byte in enumerate(value_bytes))
    header = {
        "version": PHI_ENCRYPTION_VERSION,
        "key_id": settings.encryption_key_id,
        "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    packed = len(header_bytes).to_bytes(4, "big") + header_bytes + encrypted_payload
    return base64.urlsafe_b64encode(packed).decode("ascii")


def decrypt_phi_value(encrypted_value: str, *, token: str) -> str:
    packed = base64.urlsafe_b64decode(encrypted_value.encode("ascii"))
    if len(packed) < 4:
        raise ValueError("Encrypted PHI payload is malformed")

    header_length = int.from_bytes(packed[:4], "big")
    header_start = 4
    header_end = header_start + header_length
    if header_length <= 0 or len(packed) < header_end:
        raise ValueError("Encrypted PHI header is malformed")

    header = json.loads(packed[header_start:header_end])
    if header.get("version") != PHI_ENCRYPTION_VERSION:
        raise ValueError("Unsupported PHI encryption version")
    if header.get("token_hash") != hashlib.sha256(token.encode()).hexdigest():
        raise ValueError("Encrypted PHI token does not match")

    encrypted_payload = packed[header_end:]
    stream = phi_keystream(nonce_for_token(token), len(encrypted_payload))
    value_bytes = bytes(byte ^ stream[index] for index, byte in enumerate(encrypted_payload))
    return value_bytes.decode("utf-8")


def nonce_for_token(token: str) -> bytes:
    return hmac.new(
        settings.document_storage_key.encode(),
        token.encode(),
        hashlib.sha256,
    ).digest()[:16]


def phi_keystream(nonce: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0
    key = hashlib.sha256(settings.document_storage_key.encode()).digest()
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
