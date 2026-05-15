import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import AccessPolicyEffect, enum_values
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.role import Role


class AccessPolicy(TimestampMixin, Base):
    __tablename__ = "access_policies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    role_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    effect: Mapped[AccessPolicyEffect] = mapped_column(
        SQLEnum(AccessPolicyEffect, values_callable=enum_values, name="access_policy_effect"),
        nullable=False,
    )
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    role: Mapped["Role"] = relationship(back_populates="access_policies")
