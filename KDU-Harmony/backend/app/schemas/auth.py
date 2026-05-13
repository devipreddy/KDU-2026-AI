from uuid import UUID

from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class RoleResponse(BaseModel):
    id: UUID
    name: str
    display_name: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    department: str | None
    roles: list[RoleResponse]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RbacStatusResponse(BaseModel):
    allowed: bool
    required_roles: list[str]
    user: UserResponse
