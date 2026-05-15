import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin


class PhiMapping(TimestampMixin, Base):
    __tablename__ = "phi_mappings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    patient_ref: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    token: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    encrypted_value: Mapped[str] = mapped_column(Text, nullable=False)
    encryption_key_id: Mapped[str] = mapped_column(String(160), nullable=False)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
