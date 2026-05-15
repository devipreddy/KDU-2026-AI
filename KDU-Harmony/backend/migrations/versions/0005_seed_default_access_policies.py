"""seed default document access policies

Revision ID: 0005_seed_policies
Revises: 0004_add_chunk_indexing_status
Create Date: 2026-05-13 00:00:00.000000
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_seed_policies"
down_revision: str | None = "0004_add_chunk_indexing_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

POLICY_ROWS = (
    {
        "id": "30000000-0000-4000-8000-000000000001",
        "role_name": "doctor",
        "name": "doctor_document_access",
        "description": "Default document access policy for Doctor.",
        "effect": "allow",
        "resource_type": "document",
        "conditions": {
            "phi_visibility": "full",
            "sensitivity_levels": ["low", "medium", "high", "restricted"],
            "requires_treatment_relationship": True,
        },
        "priority": 100,
        "is_active": True,
    },
    {
        "id": "30000000-0000-4000-8000-000000000002",
        "role_name": "nurse",
        "name": "nurse_document_access",
        "description": "Default document access policy for Nurse.",
        "effect": "allow",
        "resource_type": "document",
        "conditions": {
            "phi_visibility": "limited",
            "sensitivity_levels": ["low", "medium", "high"],
            "requires_care_team_assignment": True,
        },
        "priority": 100,
        "is_active": True,
    },
    {
        "id": "30000000-0000-4000-8000-000000000003",
        "role_name": "admin",
        "name": "admin_document_access",
        "description": "Default document access policy for Administrator.",
        "effect": "allow",
        "resource_type": "document",
        "conditions": {
            "phi_visibility": "metadata_only",
            "sensitivity_levels": ["low", "medium", "high", "restricted"],
        },
        "priority": 100,
        "is_active": True,
    },
    {
        "id": "30000000-0000-4000-8000-000000000004",
        "role_name": "researcher",
        "name": "researcher_document_access",
        "description": "Default document access policy for Researcher.",
        "effect": "allow",
        "resource_type": "document",
        "conditions": {
            "phi_visibility": "de_identified",
            "sensitivity_levels": ["low", "medium"],
        },
        "priority": 100,
        "is_active": True,
    },
    {
        "id": "30000000-0000-4000-8000-000000000005",
        "role_name": "records_staff",
        "name": "records_staff_document_access",
        "description": "Default document access policy for Records Staff.",
        "effect": "allow",
        "resource_type": "document",
        "conditions": {
            "phi_visibility": "operational",
            "sensitivity_levels": ["low", "medium", "high"],
        },
        "priority": 100,
        "is_active": True,
    },
)


def upgrade() -> None:
    bind = op.get_bind()
    statement = sa.text(
        """
        INSERT INTO access_policies (
            id,
            role_id,
            name,
            description,
            effect,
            resource_type,
            conditions,
            priority,
            is_active
        )
        SELECT
            CAST(:id AS uuid),
            roles.id,
            :name,
            :description,
            CAST(:effect AS access_policy_effect),
            :resource_type,
            CAST(:conditions AS json),
            :priority,
            :is_active
        FROM roles
        WHERE roles.name = CAST(:role_name AS role_name)
        ON CONFLICT (id) DO UPDATE SET
            role_id = EXCLUDED.role_id,
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            effect = EXCLUDED.effect,
            resource_type = EXCLUDED.resource_type,
            conditions = EXCLUDED.conditions,
            priority = EXCLUDED.priority,
            is_active = EXCLUDED.is_active,
            updated_at = now()
        """
    )
    bind.execute(statement, serialized_policy_rows())


def downgrade() -> None:
    policy_ids = ", ".join(f"'{row['id']}'" for row in POLICY_ROWS)
    op.execute(sa.text(f"DELETE FROM access_policies WHERE id IN ({policy_ids})"))


def serialized_policy_rows() -> list[dict[str, object]]:
    return [
        {**row, "conditions": json.dumps(row["conditions"], sort_keys=True)}
        for row in POLICY_ROWS
    ]
