import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, ForeignKey, Numeric, String, Uuid
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import DocumentStatus, DocumentType, SensitivityLevel, enum_values
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document_chunk import DocumentChunk
    from app.models.ingestion_job import IngestionJob
    from app.models.user import User


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    patient_ref: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    visit_id: Mapped[str | None] = mapped_column(String(80), index=True)
    document_type: Mapped[DocumentType] = mapped_column(
        SQLEnum(DocumentType, values_callable=enum_values, name="document_type"),
        nullable=False,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus, values_callable=enum_values, name="document_status"),
        default=DocumentStatus.UPLOADED,
        nullable=False,
        index=True,
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    hospital: Mapped[str | None] = mapped_column(String(160), index=True)
    physician: Mapped[str | None] = mapped_column(String(160), index=True)
    department: Mapped[str | None] = mapped_column(String(120), index=True)
    diagnosis: Mapped[str | None] = mapped_column(String(255), index=True)
    icd_codes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    sensitivity_level: Mapped[SensitivityLevel] = mapped_column(
        SQLEnum(SensitivityLevel, values_callable=enum_values, name="sensitivity_level"),
        default=SensitivityLevel.MEDIUM,
        nullable=False,
        index=True,
    )
    is_encrypted: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    ocr_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ocr_engine: Mapped[str | None] = mapped_column(String(80))
    ocr_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
    )
    document_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    created_by: Mapped["User | None"] = relationship(back_populates="created_documents")
