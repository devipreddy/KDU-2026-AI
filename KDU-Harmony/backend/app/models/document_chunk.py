import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Numeric, String, Text, Uuid
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import ChunkIndexingStatus, SensitivityLevel, enum_values
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(TimestampMixin, Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(120), index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    embedding_collection: Mapped[str | None] = mapped_column(String(120), index=True)
    embedding_id: Mapped[str | None] = mapped_column(String(160), unique=True)
    indexing_status: Mapped[str] = mapped_column(
        String(40),
        default=ChunkIndexingStatus.PENDING.value,
        nullable=False,
        index=True,
    )
    indexing_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    indexing_error: Mapped[str | None] = mapped_column(Text)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    token_count: Mapped[int | None] = mapped_column(Integer)
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    ocr_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4))
    sensitivity_level: Mapped[SensitivityLevel] = mapped_column(
        SQLEnum(SensitivityLevel, values_callable=enum_values, name="sensitivity_level"),
        default=SensitivityLevel.MEDIUM,
        nullable=False,
        index=True,
    )
    retrieval_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    parent_chunk: Mapped["DocumentChunk | None"] = relationship(remote_side=[id])
