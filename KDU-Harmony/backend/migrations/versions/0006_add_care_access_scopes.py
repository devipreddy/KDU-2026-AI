"""add organization scopes and patient care assignments

Revision ID: 0006_care_access_scopes
Revises: 0005_seed_policies
Create Date: 2026-05-15 00:00:00.000000
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_care_access_scopes"
down_revision: str | None = "0005_seed_policies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ORGANIZATION_SCOPE_ROWS = (
    {
        "id": "40000000-0000-4000-8000-000000000001",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "hospital": "Harmony General Hospital",
        "is_primary": True,
    },
    {
        "id": "40000000-0000-4000-8000-000000000002",
        "user_id": "20000000-0000-4000-8000-000000000002",
        "hospital": "Harmony General Hospital",
        "is_primary": True,
    },
)

PATIENT_ASSIGNMENT_ROWS = (
    {
        "id": "41000000-0000-4000-8000-000000000001",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_2301",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000002",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0300",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000003",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0100",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000004",
        "user_id": "20000000-0000-4000-8000-000000000001",
        "patient_ref": "PATIENT_REF_0001",
        "hospital": "Harmony General Hospital",
    },
    {
        "id": "41000000-0000-4000-8000-000000000005",
        "user_id": "20000000-0000-4000-8000-000000000002",
        "patient_ref": "PATIENT_REF_0042",
        "hospital": "Harmony General Hospital",
    },
)

POLICY_CONDITION_UPDATES = {
    "doctor_document_access": {
        "phi_visibility": "full",
        "sensitivity_levels": ["low", "medium", "high", "restricted"],
        "requires_organization_scope": True,
        "requires_treatment_relationship": True,
    },
    "nurse_document_access": {
        "phi_visibility": "limited",
        "sensitivity_levels": ["low", "medium", "high"],
        "requires_organization_scope": True,
        "requires_care_team_assignment": True,
    },
}


def upgrade() -> None:
    op.create_table(
        "user_organization_scopes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("hospital", sa.String(length=160), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_organization_scopes_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_organization_scopes")),
        sa.UniqueConstraint(
            "user_id",
            "hospital",
            name="uq_user_organization_scopes_user_hospital",
        ),
    )
    op.create_index(
        op.f("ix_user_organization_scopes_hospital"),
        "user_organization_scopes",
        ["hospital"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_organization_scopes_is_active"),
        "user_organization_scopes",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_organization_scopes_user_id"),
        "user_organization_scopes",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "patient_care_assignments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("patient_ref", sa.String(length=80), nullable=False),
        sa.Column("hospital", sa.String(length=160), nullable=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_patient_care_assignments_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patient_care_assignments")),
        sa.UniqueConstraint(
            "user_id",
            "patient_ref",
            "hospital",
            name="uq_patient_care_assignments_user_patient_hospital",
        ),
    )
    op.create_index(
        op.f("ix_patient_care_assignments_hospital"),
        "patient_care_assignments",
        ["hospital"],
        unique=False,
    )
    op.create_index(
        op.f("ix_patient_care_assignments_is_active"),
        "patient_care_assignments",
        ["is_active"],
        unique=False,
    )
    op.create_index(
        op.f("ix_patient_care_assignments_patient_ref"),
        "patient_care_assignments",
        ["patient_ref"],
        unique=False,
    )
    op.create_index(
        op.f("ix_patient_care_assignments_user_id"),
        "patient_care_assignments",
        ["user_id"],
        unique=False,
    )

    bind = op.get_bind()
    bind.execute(sa.text(organization_scope_insert_sql()), list(ORGANIZATION_SCOPE_ROWS))
    bind.execute(sa.text(patient_assignment_insert_sql()), list(PATIENT_ASSIGNMENT_ROWS))
    for policy_name, conditions in POLICY_CONDITION_UPDATES.items():
        bind.execute(
            sa.text(
                """
                UPDATE access_policies
                SET conditions = CAST(:conditions AS json), updated_at = now()
                WHERE name = :name
                """
            ),
            {"name": policy_name, "conditions": json.dumps(conditions, sort_keys=True)},
        )


def downgrade() -> None:
    for policy_name, conditions in {
        "doctor_document_access": {
            "phi_visibility": "full",
            "sensitivity_levels": ["low", "medium", "high", "restricted"],
            "requires_treatment_relationship": True,
        },
        "nurse_document_access": {
            "phi_visibility": "limited",
            "sensitivity_levels": ["low", "medium", "high"],
            "requires_care_team_assignment": True,
        },
    }.items():
        op.execute(
            sa.text(
                """
                UPDATE access_policies
                SET conditions = CAST(:conditions AS json), updated_at = now()
                WHERE name = :name
                """
            ).bindparams(name=policy_name, conditions=json.dumps(conditions, sort_keys=True))
        )

    op.drop_index(
        op.f("ix_patient_care_assignments_user_id"), table_name="patient_care_assignments"
    )
    op.drop_index(
        op.f("ix_patient_care_assignments_patient_ref"),
        table_name="patient_care_assignments",
    )
    op.drop_index(
        op.f("ix_patient_care_assignments_is_active"),
        table_name="patient_care_assignments",
    )
    op.drop_index(
        op.f("ix_patient_care_assignments_hospital"), table_name="patient_care_assignments"
    )
    op.drop_table("patient_care_assignments")
    op.drop_index(
        op.f("ix_user_organization_scopes_user_id"), table_name="user_organization_scopes"
    )
    op.drop_index(
        op.f("ix_user_organization_scopes_is_active"),
        table_name="user_organization_scopes",
    )
    op.drop_index(
        op.f("ix_user_organization_scopes_hospital"),
        table_name="user_organization_scopes",
    )
    op.drop_table("user_organization_scopes")


def organization_scope_insert_sql() -> str:
    return """
        INSERT INTO user_organization_scopes (
            id, user_id, hospital, is_primary, is_active, created_at, updated_at
        )
        VALUES (
            CAST(:id AS uuid), CAST(:user_id AS uuid), :hospital, :is_primary, true, now(), now()
        )
        ON CONFLICT (user_id, hospital) DO UPDATE SET
            is_primary = EXCLUDED.is_primary,
            is_active = true,
            updated_at = now()
    """


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
