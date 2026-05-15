from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_current_user, require_roles, role_names_for
from app.core.config import settings
from app.core.security import create_access_token, verify_password
from app.db.session import get_db
from app.models.audit_event import AuditEvent
from app.models.enums import AuditAction, RoleName
from app.models.user import User
from app.schemas.auth import LoginRequest, RbacStatusResponse, TokenResponse, UserResponse

router = APIRouter(prefix="/auth")
DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
RequireAdmin = require_roles(RoleName.ADMIN.value)
AdminUser = Annotated[User, Depends(RequireAdmin)]


def serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        department=user.department,
        roles=[
            {
                "id": role.id,
                "name": role.name.value,
                "display_name": role.display_name,
            }
            for role in sorted(user.roles, key=lambda item: item.name.value)
        ],
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    user = db.scalar(
        select(User)
        .options(selectinload(User.roles))
        .where(func.lower(User.email) == payload.email.lower())
    )
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role_names = role_names_for(user)
    user.last_login_at = datetime.now(UTC)
    db.add(
        AuditEvent(
            actor=user,
            action=AuditAction.LOGIN,
            decision="allow",
            role_snapshot=role_names,
            event_metadata={"auth_method": "password"},
        )
    )
    db.commit()

    token = create_access_token(subject=str(user.id), roles=role_names)
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=serialize_user(user),
    )


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: CurrentUser) -> UserResponse:
    return serialize_user(current_user)


@router.get("/rbac/admin", response_model=RbacStatusResponse)
def read_admin_rbac_status(current_user: AdminUser) -> RbacStatusResponse:
    return RbacStatusResponse(
        allowed=True,
        required_roles=[RoleName.ADMIN.value],
        user=serialize_user(current_user),
    )
