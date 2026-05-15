"""create core healthcare search tables

Revision ID: 0001_create_core_tables
Revises:
Create Date: 2026-05-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_create_core_tables"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

role_name = postgresql.ENUM(
    "doctor",
    "nurse",
    "admin",
    "researcher",
    "records_staff",
    "billing",
    "system",
    name="role_name",
    create_type=False,
)
document_type = postgresql.ENUM(
    "typed_pdf",
    "scanned_pdf",
    "clinical_note",
    "discharge_summary",
    "lab_report",
    "prescription",
    "handwritten_note",
    "other",
    name="document_type",
    create_type=False,
)
document_status = postgresql.ENUM(
    "uploaded",
    "extracting",
    "processed",
    "indexed",
    "failed",
    "archived",
    name="document_status",
    create_type=False,
)
sensitivity_level = postgresql.ENUM(
    "low",
    "medium",
    "high",
    "restricted",
    name="sensitivity_level",
    create_type=False,
)
ingestion_job_status = postgresql.ENUM(
    "queued",
    "running",
    "succeeded",
    "failed",
    "dead_letter",
    name="ingestion_job_status",
    create_type=False,
)
audit_action = postgresql.ENUM(
    "login",
    "document_upload",
    "document_read",
    "query_run",
    "phi_decrypt",
    "break_glass",
    "access_denied",
    name="audit_action",
    create_type=False,
)
access_policy_effect = postgresql.ENUM(
    "allow",
    "deny",
    name="access_policy_effect",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    for enum_type in (
        role_name,
        document_type,
        document_status,
        sensitivity_level,
        ingestion_job_status,
        audit_action,
        access_policy_effect,
    ):
        enum_type.create(bind, checkfirst=True)

    op.create_table(
        "roles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", role_name, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_roles")),
        sa.UniqueConstraint("name", name=op.f("uq_roles_name")),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=160), nullable=False),
        sa.Column("department", sa.String(length=120), nullable=True),
        sa.Column("external_subject", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
        sa.UniqueConstraint("external_subject", name=op.f("uq_users_external_subject")),
    )
    op.create_index(op.f("ix_users_department"), "users", ["department"], unique=False)
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
    op.create_index(op.f("ix_users_external_subject"), "users", ["external_subject"], unique=False)

    op.create_table(
        "user_roles",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["role_id"], ["roles.id"], name=op.f("fk_user_roles_role_id_roles"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_user_roles_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "role_id", name=op.f("pk_user_roles")),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=80), nullable=False),
        sa.Column("patient_ref", sa.String(length=80), nullable=False),
        sa.Column("visit_id", sa.String(length=80), nullable=True),
        sa.Column("document_type", document_type, nullable=False),
        sa.Column("status", document_status, nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("source_uri", sa.String(length=500), nullable=False),
        sa.Column("mime_type", sa.String(length=120), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("hospital", sa.String(length=160), nullable=True),
        sa.Column("physician", sa.String(length=160), nullable=True),
        sa.Column("department", sa.String(length=120), nullable=True),
        sa.Column("diagnosis", sa.String(length=255), nullable=True),
        sa.Column("icd_codes", sa.JSON(), nullable=False),
        sa.Column("sensitivity_level", sensitivity_level, nullable=False),
        sa.Column("is_encrypted", sa.Boolean(), nullable=False),
        sa.Column("ocr_required", sa.Boolean(), nullable=False),
        sa.Column("ocr_engine", sa.String(length=80), nullable=True),
        sa.Column("ocr_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name=op.f("fk_documents_created_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("external_id", name=op.f("uq_documents_external_id")),
    )
    op.create_index(
        op.f("ix_documents_checksum_sha256"), "documents", ["checksum_sha256"], unique=False
    )
    op.create_index(
        op.f("ix_documents_created_by_user_id"), "documents", ["created_by_user_id"], unique=False
    )
    op.create_index(op.f("ix_documents_department"), "documents", ["department"], unique=False)
    op.create_index(op.f("ix_documents_diagnosis"), "documents", ["diagnosis"], unique=False)
    op.create_index(op.f("ix_documents_external_id"), "documents", ["external_id"], unique=False)
    op.create_index(op.f("ix_documents_hospital"), "documents", ["hospital"], unique=False)
    op.create_index(op.f("ix_documents_patient_ref"), "documents", ["patient_ref"], unique=False)
    op.create_index(op.f("ix_documents_physician"), "documents", ["physician"], unique=False)
    op.create_index(
        op.f("ix_documents_sensitivity_level"), "documents", ["sensitivity_level"], unique=False
    )
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)
    op.create_index(op.f("ix_documents_visit_id"), "documents", ["visit_id"], unique=False)

    op.create_table(
        "access_policies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("role_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("effect", access_policy_effect, nullable=False),
        sa.Column("resource_type", sa.String(length=80), nullable=False),
        sa.Column("conditions", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["role_id"],
            ["roles.id"],
            name=op.f("fk_access_policies_role_id_roles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_access_policies")),
    )
    op.create_index(
        op.f("ix_access_policies_role_id"), "access_policies", ["role_id"], unique=False
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("action", audit_action, nullable=False),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("resource_type", sa.String(length=80), nullable=True),
        sa.Column("resource_id", sa.Uuid(), nullable=True),
        sa.Column("result_document_ids", sa.JSON(), nullable=False),
        sa.Column("decision", sa.String(length=40), nullable=True),
        sa.Column("role_snapshot", sa.JSON(), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name=op.f("fk_audit_events_actor_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_events")),
    )
    op.create_index(op.f("ix_audit_events_action"), "audit_events", ["action"], unique=False)
    op.create_index(
        op.f("ix_audit_events_actor_user_id"), "audit_events", ["actor_user_id"], unique=False
    )
    op.create_index(
        op.f("ix_audit_events_resource_id"), "audit_events", ["resource_id"], unique=False
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("parent_chunk_id", sa.Uuid(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=120), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("embedding_collection", sa.String(length=120), nullable=True),
        sa.Column("embedding_id", sa.String(length=160), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("ocr_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("sensitivity_level", sensitivity_level, nullable=False),
        sa.Column("retrieval_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_chunk_id"],
            ["document_chunks.id"],
            name=op.f("fk_document_chunks_parent_chunk_id_document_chunks"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint("embedding_id", name=op.f("uq_document_chunks_embedding_id")),
    )
    op.create_index(
        op.f("ix_document_chunks_content_sha256"),
        "document_chunks",
        ["content_sha256"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"), "document_chunks", ["document_id"], unique=False
    )
    op.create_index(
        op.f("ix_document_chunks_embedding_collection"),
        "document_chunks",
        ["embedding_collection"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_parent_chunk_id"),
        "document_chunks",
        ["parent_chunk_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_section"), "document_chunks", ["section"], unique=False
    )
    op.create_index(
        op.f("ix_document_chunks_sensitivity_level"),
        "document_chunks",
        ["sensitivity_level"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_document_chunks_document_id_chunk_index",
        "document_chunks",
        ["document_id", "chunk_index"],
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("status", ingestion_job_status, nullable=False),
        sa.Column("stage", sa.String(length=80), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("job_metadata", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_ingestion_jobs_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
    )
    op.create_index(
        op.f("ix_ingestion_jobs_document_id"), "ingestion_jobs", ["document_id"], unique=False
    )
    op.create_index(op.f("ix_ingestion_jobs_status"), "ingestion_jobs", ["status"], unique=False)

    op.create_table(
        "phi_mappings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("patient_ref", sa.String(length=80), nullable=False),
        sa.Column("token", sa.String(length=120), nullable=False),
        sa.Column("entity_type", sa.String(length=80), nullable=False),
        sa.Column("encrypted_value", sa.Text(), nullable=False),
        sa.Column("encryption_key_id", sa.String(length=160), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name=op.f("fk_phi_mappings_created_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_phi_mappings")),
        sa.UniqueConstraint("token", name=op.f("uq_phi_mappings_token")),
    )
    op.create_index(
        op.f("ix_phi_mappings_created_by_user_id"),
        "phi_mappings",
        ["created_by_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_phi_mappings_entity_type"), "phi_mappings", ["entity_type"], unique=False
    )
    op.create_index(
        op.f("ix_phi_mappings_patient_ref"), "phi_mappings", ["patient_ref"], unique=False
    )
    op.create_index(op.f("ix_phi_mappings_token"), "phi_mappings", ["token"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_phi_mappings_token"), table_name="phi_mappings")
    op.drop_index(op.f("ix_phi_mappings_patient_ref"), table_name="phi_mappings")
    op.drop_index(op.f("ix_phi_mappings_entity_type"), table_name="phi_mappings")
    op.drop_index(op.f("ix_phi_mappings_created_by_user_id"), table_name="phi_mappings")
    op.drop_table("phi_mappings")

    op.drop_index(op.f("ix_ingestion_jobs_status"), table_name="ingestion_jobs")
    op.drop_index(op.f("ix_ingestion_jobs_document_id"), table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_constraint(
        "uq_document_chunks_document_id_chunk_index", "document_chunks", type_="unique"
    )
    op.drop_index(op.f("ix_document_chunks_sensitivity_level"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_section"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_parent_chunk_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_embedding_collection"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_content_sha256"), table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_index(op.f("ix_audit_events_resource_id"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_actor_user_id"), table_name="audit_events")
    op.drop_index(op.f("ix_audit_events_action"), table_name="audit_events")
    op.drop_table("audit_events")

    op.drop_index(op.f("ix_access_policies_role_id"), table_name="access_policies")
    op.drop_table("access_policies")

    op.drop_index(op.f("ix_documents_visit_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index(op.f("ix_documents_sensitivity_level"), table_name="documents")
    op.drop_index(op.f("ix_documents_physician"), table_name="documents")
    op.drop_index(op.f("ix_documents_patient_ref"), table_name="documents")
    op.drop_index(op.f("ix_documents_hospital"), table_name="documents")
    op.drop_index(op.f("ix_documents_external_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_diagnosis"), table_name="documents")
    op.drop_index(op.f("ix_documents_department"), table_name="documents")
    op.drop_index(op.f("ix_documents_created_by_user_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_checksum_sha256"), table_name="documents")
    op.drop_table("documents")

    op.drop_table("user_roles")

    op.drop_index(op.f("ix_users_external_subject"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_department"), table_name="users")
    op.drop_table("users")

    op.drop_table("roles")

    bind = op.get_bind()
    for enum_type in (
        access_policy_effect,
        audit_action,
        ingestion_job_status,
        sensitivity_level,
        document_status,
        document_type,
        role_name,
    ):
        enum_type.drop(bind, checkfirst=True)
