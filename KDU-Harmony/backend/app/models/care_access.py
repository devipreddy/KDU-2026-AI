import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class UserOrganizationScope(TimestampMixin, Base):
    __tablename__ = "user_organization_scopes"
    __table_args__ = (
        UniqueConstraint("user_id", "hospital", name="uq_user_organization_scopes_user_hospital"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    hospital: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    user: Mapped["User"] = relationship(back_populates="organization_scopes")


class PatientCareAssignment(TimestampMixin, Base):
    __tablename__ = "patient_care_assignments"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "patient_ref",
            "hospital",
            name="uq_patient_care_assignments_user_patient_hospital",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    patient_ref: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    hospital: Mapped[str | None] = mapped_column(String(160), index=True)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    user: Mapped["User"] = relationship(back_populates="patient_assignments")
