"""add chunk indexing status tracking

Revision ID: 0004_add_chunk_indexing_status
Revises: 0003_add_document_review_status
Create Date: 2026-05-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_add_chunk_indexing_status"
down_revision: str | None = "0003_add_document_review_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "indexing_status",
            sa.String(length=40),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "document_chunks",
        sa.Column("indexing_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )
    op.add_column("document_chunks", sa.Column("indexing_error", sa.Text(), nullable=True))
    op.add_column(
        "document_chunks",
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_document_chunks_indexing_status"),
        "document_chunks",
        ["indexing_status"],
        unique=False,
    )
    op.alter_column("document_chunks", "indexing_status", server_default=None)
    op.alter_column("document_chunks", "indexing_attempts", server_default=None)


def downgrade() -> None:
    op.drop_index(op.f("ix_document_chunks_indexing_status"), table_name="document_chunks")
    op.drop_column("document_chunks", "indexed_at")
    op.drop_column("document_chunks", "indexing_error")
    op.drop_column("document_chunks", "indexing_attempts")
    op.drop_column("document_chunks", "indexing_status")
