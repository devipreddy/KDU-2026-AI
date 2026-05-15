from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import settings

JWT_ALGORITHM = "HS256"
PASSWORD_SCHEME = "scrypt"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _base64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(f"{encoded}{padding}")


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return (
        f"{PASSWORD_SCHEME}$n={SCRYPT_N}$r={SCRYPT_R}$p={SCRYPT_P}"
        f"${_base64url_encode(salt)}${_base64url_encode(digest)}"
    )


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False

    try:
        scheme, n_part, r_part, p_part, salt_part, digest_part = stored_hash.split("$")
        if scheme != PASSWORD_SCHEME:
            return False
        n = int(n_part.removeprefix("n="))
        r = int(r_part.removeprefix("r="))
        p = int(p_part.removeprefix("p="))
        salt = _base64url_decode(salt_part)
        expected_digest = _base64url_decode(digest_part)
    except (ValueError, TypeError):
        return False

    actual_digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected_digest),
    )
    return hmac.compare_digest(actual_digest, expected_digest)


def create_access_token(
    *,
    subject: str,
    roles: list[str],
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(UTC)
    expires_at = now + (
        expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes)
    )
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "iss": settings.jwt_issuer,
        "sub": subject,
        "roles": roles,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    signing_input = ".".join(
        (
            _base64url_encode(_json_dumps(header)),
            _base64url_encode(_json_dumps(payload)),
        )
    )
    signature = hmac.new(
        settings.jwt_secret_key.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_base64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = json.loads(_base64url_decode(encoded_header))
        payload = json.loads(_base64url_decode(encoded_payload))
        signature = _base64url_decode(encoded_signature)
    except (ValueError, json.JSONDecodeError):
        raise ValueError("Invalid token format") from None

    if header.get("alg") != JWT_ALGORITHM:
        raise ValueError("Unsupported token algorithm")

    signing_input = f"{encoded_header}.{encoded_payload}"
    expected_signature = hmac.new(
        settings.jwt_secret_key.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid token signature")

    expires_at = payload.get("exp")
    if not isinstance(expires_at, int):
        raise ValueError("Token is missing expiration")
    if datetime.now(UTC).timestamp() >= expires_at:
        raise ValueError("Token has expired")

    if payload.get("iss") != settings.jwt_issuer:
        raise ValueError("Invalid token issuer")

    return payload
