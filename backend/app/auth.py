from __future__ import annotations

import hmac
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, Field

from .config import Settings
from .context import ActorRole, RequestContext


class SessionEnvelope(BaseModel):
    client_secret: str
    expires_at: datetime
    user_id: str
    session_id: str
    role: ActorRole


class ClientSecretClaims(BaseModel):
    sub: str
    sid: str
    role: ActorRole
    jti: str
    iat: int
    nbf: int
    exp: int
    display_name: str | None = None


@dataclass(slots=True)
class BrowserIdentity:
    user_id: str
    session_id: str


class AuthError(HTTPException):
    def __init__(self, detail: str, status_code: int = 401):
        super().__init__(status_code=status_code, detail=detail)


class AuthService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def ensure_customer_identity(
        self, request: Request, response: Response
    ) -> BrowserIdentity:
        user_id = request.cookies.get(self.settings.customer_cookie_name)
        session_id = request.cookies.get(self.settings.session_cookie_name)

        if not user_id:
            user_id = f"user_{uuid.uuid4().hex}"
            response.set_cookie(
                self.settings.customer_cookie_name,
                user_id,
                httponly=True,
                samesite="lax",
                secure=False,
                max_age=60 * 60 * 24 * 30,
            )

        if not session_id:
            session_id = f"sid_{uuid.uuid4().hex}"
            response.set_cookie(
                self.settings.session_cookie_name,
                session_id,
                httponly=True,
                samesite="lax",
                secure=False,
                max_age=60 * 60 * 24 * 7,
            )

        return BrowserIdentity(user_id=user_id, session_id=session_id)

    def issue_client_secret(
        self,
        *,
        user_id: str,
        session_id: str,
        role: ActorRole,
        display_name: str | None = None,
    ) -> SessionEnvelope:
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.settings.client_secret_ttl_seconds)
        claims = ClientSecretClaims(
            sub=user_id,
            sid=session_id,
            role=role,
            display_name=display_name,
            jti=f"jti_{uuid.uuid4().hex}",
            iat=int(now.timestamp()),
            nbf=int(now.timestamp()),
            exp=int(expires_at.timestamp()),
        )
        token = jwt.encode(
            claims.model_dump(),
            self.settings.jwt_secret.get_secret_value(),
            algorithm=self.settings.jwt_algorithm,
        )
        return SessionEnvelope(
            client_secret=token,
            expires_at=expires_at,
            user_id=user_id,
            session_id=session_id,
            role=role,
        )

    def validate_client_secret(
        self,
        token: str,
        *,
        expected_session_id: str | None = None,
        expected_role: ActorRole | None = None,
    ) -> RequestContext:
        try:
            raw_claims = jwt.decode(
                token,
                self.settings.jwt_secret.get_secret_value(),
                algorithms=[self.settings.jwt_algorithm],
            )
        except jwt.InvalidTokenError as exc:
            raise AuthError("Invalid or expired client_secret") from exc

        claims = ClientSecretClaims.model_validate(raw_claims)

        if expected_role and claims.role != expected_role:
            raise AuthError("This client_secret is not valid for the requested role", 403)

        if claims.role == "customer" and expected_session_id:
            if not hmac.compare_digest(expected_session_id, claims.sid):
                raise AuthError("Session cookie and client_secret do not match")

        return RequestContext(
            user_id=claims.sub,
            session_id=claims.sid,
            role=claims.role,
            display_name=claims.display_name,
        )

    def require_request_context(self, request: Request) -> RequestContext:
        token = self._extract_bearer_token(request.headers)
        expected_session_id = request.cookies.get(self.settings.session_cookie_name)
        context = self.validate_client_secret(
            token,
            expected_session_id=expected_session_id,
        )
        return RequestContext(
            user_id=context.user_id,
            session_id=context.session_id,
            role=context.role,
            request_id=f"req_{uuid.uuid4().hex[:12]}",
            display_name=context.display_name,
        )

    async def require_websocket_context(
        self,
        websocket: WebSocket,
        *,
        expected_role: ActorRole | None = None,
    ) -> RequestContext:
        token = websocket.query_params.get("client_secret")
        if not token:
            raise AuthError("client_secret query parameter is required")
        expected_session_id = websocket.cookies.get(self.settings.session_cookie_name)
        context = self.validate_client_secret(
            token,
            expected_session_id=expected_session_id,
            expected_role=expected_role,
        )
        return RequestContext(
            user_id=context.user_id,
            session_id=context.session_id,
            role=context.role,
            request_id=f"ws_{uuid.uuid4().hex[:12]}",
            display_name=context.display_name,
        )

    def validate_agent_dashboard_token(self, token: str | None) -> None:
        if not token:
            raise AuthError("Missing agent dashboard token")
        if not hmac.compare_digest(
            token,
            self.settings.human_agent_dashboard_token.get_secret_value(),
        ):
            raise AuthError("Invalid agent dashboard token", 403)

    def _extract_bearer_token(self, headers: Any) -> str:
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if not auth_header:
            raise AuthError("Missing Authorization header")

        prefix = "Bearer "
        if not auth_header.startswith(prefix):
            raise AuthError("Authorization header must use Bearer auth")
        return auth_header[len(prefix) :].strip()

