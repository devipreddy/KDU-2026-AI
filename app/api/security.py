from __future__ import annotations

from fastapi import Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    bearer_credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    configured_api_key = get_settings().app_api_key
    if not configured_api_key:
        return

    provided_token = x_api_key or (bearer_credentials.credentials if bearer_credentials else None)
    if provided_token != configured_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
