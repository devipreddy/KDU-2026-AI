"""seed auth roles and demo users

Revision ID: 0002_seed_auth_rbac
Revises: 0001_create_core_tables
Create Date: 2026-05-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_seed_auth_rbac"
down_revision: str | None = "0001_create_core_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEMO_ROLE_ROWS = (
    {
        "id": "10000000-0000-4000-8000-000000000001",
        "name": "doctor",
        "display_name": "Doctor",
        "description": "Treating clinician with full clinical context for authorized patients.",
    },
    {
        "id": "10000000-0000-4000-8000-000000000002",
        "name": "nurse",
        "display_name": "Nurse",
        "description": "Care team user with limited clinical context.",
    },
    {
        "id": "10000000-0000-4000-8000-000000000003",
        "name": "admin",
        "display_name": "Administrator",
        "description": "Platform administrator with metadata and system-management access.",
    },
    {
        "id": "10000000-0000-4000-8000-000000000004",
        "name": "researcher",
        "display_name": "Researcher",
        "description": "Research user restricted to de-identified output.",
    },
    {
        "id": "10000000-0000-4000-8000-000000000005",
        "name": "records_staff",
        "display_name": "Records Staff",
        "description": "Records operations user with document workflow access.",
    },
)

DEMO_USER_ROWS = (
    {
        "id": "20000000-0000-4000-8000-000000000001",
        "email": "doctor@example.com",
        "display_name": "Dr. Demo Doctor",
        "department": "Cardiology",
        "external_subject": "local-dev:doctor@example.com",
        "password_hash": (
            "scrypt$n=16384$r=8$p=1$bG9jYWwtZGV2LWF1dGg6ZG9jdG9y"
            "$gMeFVlYTtDww_vlhnR-24OEyLuGpYWzC3N4SFLkFSCM"
        ),
        "is_active": True,
    },
    {
        "id": "20000000-0000-4000-8000-000000000002",
        "email": "nurse@example.com",
        "display_name": "Nora Demo Nurse",
        "department": "Care Team",
        "external_subject": "local-dev:nurse@example.com",
        "password_hash": (
            "scrypt$n=16384$r=8$p=1$bG9jYWwtZGV2LWF1dGg6bnVyc2U"
            "$KWsaDMSkADpgHjv-f69pmHXZE_MC1a_F2H3K9Q8JFrc"
        ),
        "is_active": True,
    },
    {
        "id": "20000000-0000-4000-8000-000000000003",
        "email": "admin@example.com",
        "display_name": "Ari Demo Admin",
        "department": "Platform Operations",
        "external_subject": "local-dev:admin@example.com",
        "password_hash": (
            "scrypt$n=16384$r=8$p=1$bG9jYWwtZGV2LWF1dGg6YWRtaW4"
            "$M3FkkwTyng2vhKsDLYy3AeAM5WQ6h95fgh-NxHxiXDc"
        ),
        "is_active": True,
    },
    {
        "id": "20000000-0000-4000-8000-000000000004",
        "email": "researcher@example.com",
        "display_name": "Rhea Demo Researcher",
        "department": "Clinical Research",
        "external_subject": "local-dev:researcher@example.com",
        "password_hash": (
            "scrypt$n=16384$r=8$p=1$bG9jYWwtZGV2LWF1dGg6cmVzZWFyY2hlcg"
            "$Jc3ydODy1aPIYSco2AaSOfBveGzNecy8rx8mUot-TFc"
        ),
        "is_active": True,
    },
    {
        "id": "20000000-0000-4000-8000-000000000005",
        "email": "records@example.com",
        "display_name": "Remy Demo Records",
        "department": "Records Management",
        "external_subject": "local-dev:records@example.com",
        "password_hash": (
            "scrypt$n=16384$r=8$p=1$bG9jYWwtZGV2LWF1dGg6cmVjb3Jkc19zdGFmZg"
            "$P2JsHPAUl2dEg0JNq38aDbfKgjpT8U_iLIuOONqVaxI"
        ),
        "is_active": True,
    },
)

DEMO_USER_ROLE_ROWS = (
    {
        "user_id": "20000000-0000-4000-8000-000000000001",
        "role_id": "10000000-0000-4000-8000-000000000001",
    },
    {
        "user_id": "20000000-0000-4000-8000-000000000002",
        "role_id": "10000000-0000-4000-8000-000000000002",
    },
    {
        "user_id": "20000000-0000-4000-8000-000000000003",
        "role_id": "10000000-0000-4000-8000-000000000003",
    },
    {
        "user_id": "20000000-0000-4000-8000-000000000004",
        "role_id": "10000000-0000-4000-8000-000000000004",
    },
    {
        "user_id": "20000000-0000-4000-8000-000000000005",
        "role_id": "10000000-0000-4000-8000-000000000005",
    },
)


def upgrade() -> None:
    op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))

    roles = sa.table(
        "roles",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("display_name", sa.String()),
    )
    users = sa.table(
        "users",
        sa.column("id", sa.Uuid()),
        sa.column("email", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("department", sa.String()),
        sa.column("external_subject", sa.String()),
        sa.column("password_hash", sa.String()),
        sa.column("is_active", sa.Boolean()),
    )
    user_roles = sa.table(
        "user_roles",
        sa.column("user_id", sa.Uuid()),
        sa.column("role_id", sa.Uuid()),
    )

    op.bulk_insert(roles, list(DEMO_ROLE_ROWS))
    op.bulk_insert(users, list(DEMO_USER_ROWS))
    op.bulk_insert(user_roles, list(DEMO_USER_ROLE_ROWS))


def downgrade() -> None:
    role_ids = ", ".join(f"'{row['id']}'" for row in DEMO_ROLE_ROWS)
    user_ids = ", ".join(f"'{row['id']}'" for row in DEMO_USER_ROWS)
    op.execute(sa.text(f"DELETE FROM user_roles WHERE user_id IN ({user_ids})"))
    op.execute(sa.text(f"DELETE FROM users WHERE id IN ({user_ids})"))
    op.execute(sa.text(f"DELETE FROM roles WHERE id IN ({role_ids})"))
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "password_hash")
