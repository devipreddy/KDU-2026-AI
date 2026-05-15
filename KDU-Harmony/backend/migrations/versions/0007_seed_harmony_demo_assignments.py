"""seed harmony demo patient assignments

Revision ID: 0007_seed_harmony_assignments
Revises: 0006_care_access_scopes
Create Date: 2026-05-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_seed_harmony_assignments"
down_revision: str | None = "0006_care_access_scopes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEMO_ASSIGNMENT_ROWS = (
    {
        "id": "41000000-0000-4000-8000-000000000006",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0050",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000007",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0201",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000008",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0012",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000009",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0229",
        "hospital": "Harmony General Hospital",
    },
)


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text(patient_assignment_insert_sql()), list(DEMO_ASSIGNMENT_ROWS))


def downgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM patient_care_assignments
            WHERE id IN (
                CAST('41000000-0000-4000-8000-000000000006' AS uuid),
                CAST('41000000-0000-4000-8000-000000000007' AS uuid),
                CAST('41000000-0000-4000-8000-000000000008' AS uuid),
                CAST('41000000-0000-4000-8000-000000000009' AS uuid)
            )
            """
        )
    )


def patient_assignment_insert_sql() -> str:
    return """
        INSERT INTO patient_care_assignments (
            id,
            user_id,
            patient_ref,
            hospital,
            assigned_at,
            expires_at,
            is_active,
            created_at,
            updated_at
        )
        VALUES (
            CAST(:id AS uuid),
            CAST(:user_id AS uuid),
            :patient_ref,
            :hospital,
            now(),
            NULL,
            true,
            now(),
            now()
        )
        ON CONFLICT (user_id, patient_ref, hospital) DO UPDATE SET
            is_active = true,
            expires_at = NULL,
            updated_at = now()
    """
