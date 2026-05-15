"""add document review required status

Revision ID: 0003_add_document_review_status
Revises: 0002_seed_auth_rbac
Create Date: 2026-05-13 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_add_document_review_status"
down_revision: str | None = "0002_seed_auth_rbac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'review_required'")


def downgrade() -> None:
    # PostgreSQL cannot safely drop enum values without recreating the type.
    pass
