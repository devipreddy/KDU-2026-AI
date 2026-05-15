import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SQLEnum
from sqlalchemy import String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import RoleName, enum_values
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.access_policy import AccessPolicy
    from app.models.user import User


class Role(TimestampMixin, Base):
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[RoleName] = mapped_column(
        SQLEnum(RoleName, values_callable=enum_values, name="role_name"),
        unique=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)

    users: Mapped[list["User"]] = relationship(
        secondary="user_roles",
        back_populates="roles",
    )
    access_policies: Mapped[list["AccessPolicy"]] = relationship(
        back_populates="role",
        cascade="all, delete-orphan",
    )
