from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.role import Role
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)
BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]
DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    credentials: BearerCredentials,
    db: DbSession,
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing subject",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = UUID(str(subject))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is invalid",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = db.scalar(select(User).options(selectinload(User.roles)).where(User.id == user_id))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive or no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_roles(*required_roles: str) -> Callable[[User], User]:
    def dependency(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        user_roles = {role.name.value for role in current_user.roles}
        if not user_roles.intersection(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role permissions",
            )
        return current_user

    return dependency


def role_names_for(user: User) -> list[str]:
    return sorted(role.name.value if isinstance(role, Role) else str(role) for role in user.roles)
