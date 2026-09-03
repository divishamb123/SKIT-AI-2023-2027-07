"""Initial migration

Revision ID: 001_initial_migration
Revises:
Create Date: 2026-06-17 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial_migration"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create custom postgres enum types safely
    op.execute(
        "DO $$ BEGIN CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled'); EXCEPTION WHEN duplicate_object THEN null; END $$;"
    )
    op.execute(
        "DO $$ BEGIN CREATE TYPE modality_type AS ENUM ('text', 'image', 'audio', 'video', 'multimodal'); EXCEPTION WHEN duplicate_object THEN null; END $$;"
    )
    op.execute(
        "DO $$ BEGIN CREATE TYPE report_type AS ENUM ('pdf', 'json', 'html'); EXCEPTION WHEN duplicate_object THEN null; END $$;"
    )
    op.execute(
        "DO $$ BEGIN CREATE TYPE report_status AS ENUM ('pending', 'generating', 'completed', 'failed'); EXCEPTION WHEN duplicate_object THEN null; END $$;"
    )

    # 2. Create tables
    # users
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
        sa.CheckConstraint(
            "role IN ('user', 'admin', 'analyst')", name=op.f("ck_users_chk_user_role")
        ),
    )

    # jobs
    op.create_table(
        "jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "queued",
                "processing",
                "completed",
                "failed",
                "cancelled",
                name="job_status",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "modality",
            postgresql.ENUM(
                "text",
                "image",
                "audio",
                "video",
                "multimodal",
                name="modality_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("input_type", sa.String(length=10), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=True),
        sa.Column("input_object_key", sa.String(length=1024), nullable=True),
        sa.Column("file_name", sa.String(length=512), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_mime_type", sa.String(length=128), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("priority", sa.SmallInteger(), server_default="5", nullable=False),
        sa.Column("retry_count", sa.SmallInteger(), server_default="0", nullable=False),
        sa.Column("max_retries", sa.SmallInteger(), server_default="3", nullable=False),
        sa.Column("queue_name", sa.String(length=64), nullable=True),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_jobs_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.CheckConstraint(
            "(input_type = 'text' AND input_text IS NOT NULL AND input_object_key IS NULL) OR (input_type = 'file' AND input_object_key IS NOT NULL AND input_text IS NULL)",
            name=op.f("ck_jobs_chk_job_input"),
        ),
        sa.CheckConstraint(
            "priority BETWEEN 1 AND 10", name=op.f("ck_jobs_chk_job_priority")
        ),
        sa.CheckConstraint(
            "input_type IN ('text', 'file')", name=op.f("ck_jobs_chk_job_input_type")
        ),
    )

    # reports
    op.create_table(
        "reports",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=False),
        sa.Column(
            "report_type",
            postgresql.ENUM(
                "pdf", "json", "html", name="report_type", create_type=False
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending",
                "generating",
                "completed",
                "failed",
                name="report_status",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("minio_object_key", sa.String(length=1024), nullable=True),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("signed_url", sa.Text(), nullable=True),
        sa.Column("signed_url_expires", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_reports_job_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        sa.UniqueConstraint("job_id", "report_type", name="uq_reports_job_report_type"),
    )

    # 3. Create indexes
    op.create_index(
        "idx_users_email_lower", "users", [sa.text("LOWER(email)")], unique=True
    )
    op.create_index("idx_jobs_user_id", "jobs", ["user_id"], unique=False)
    op.create_index(
        "idx_jobs_status",
        "jobs",
        ["status"],
        unique=False,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("idx_reports_job_id", "reports", ["job_id"], unique=False)


def downgrade() -> None:
    # Drop tables
    op.drop_table("reports")
    op.drop_table("jobs")
    op.drop_table("users")

    # Drop custom enum types
    op.execute("DROP TYPE IF EXISTS report_status;")
    op.execute("DROP TYPE IF EXISTS report_type;")
    op.execute("DROP TYPE IF EXISTS modality_type;")
    op.execute("DROP TYPE IF EXISTS job_status;")
