import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.audit_event import AuditEvent
    from app.models.care_access import PatientCareAssignment, UserOrganizationScope
    from app.models.document import Document
    from app.models.role import Role


user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    department: Mapped[str | None] = mapped_column(String(120), index=True)
    external_subject: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    roles: Mapped[list["Role"]] = relationship(
        secondary=user_roles,
        back_populates="users",
    )
    created_documents: Mapped[list["Document"]] = relationship(back_populates="created_by")
    audit_events: Mapped[list["AuditEvent"]] = relationship(back_populates="actor")
    organization_scopes: Mapped[list["UserOrganizationScope"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    patient_assignments: Mapped[list["PatientCareAssignment"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
