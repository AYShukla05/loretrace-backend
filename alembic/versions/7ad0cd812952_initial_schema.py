"""initial schema

Revision ID: 7ad0cd812952
Revises:
Create Date: 2026-07-30 01:08:51.709574

"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7ad0cd812952"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536

source_type_enum = sa.Enum(
    "gutenberg_text", "wikisource", "wikipedia", "manual_upload",
    name="sourcetype", native_enum=False,
)
source_status_enum = sa.Enum(
    "pending", "scraping", "completed", "failed",
    name="sourcestatus", native_enum=False,
)
scrape_job_status_enum = sa.Enum(
    "pending", "running", "completed", "failed",
    name="scrapejobstatus", native_enum=False,
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "admins",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_admins_email", "admins", ["email"], unique=True)

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("source_type", source_type_enum, nullable=False),
        sa.Column("tradition", sa.String(50), nullable=False),
        sa.Column("status", source_status_enum, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("etag", sa.String(255), nullable=True),
        sa.Column("last_modified", sa.String(255), nullable=True),
        sa.Column("last_scraped_at", sa.DateTime(), nullable=True),
        sa.Column("last_checked_at", sa.DateTime(), nullable=True),
        sa.Column("added_by", sa.Integer(), sa.ForeignKey("admins.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_sources_url", "sources", ["url"], unique=True)
    op.create_index("ix_sources_tradition", "sources", ["tradition"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chunks_source_id", "chunks", ["source_id"])
    op.create_index("ix_chunks_chunk_hash", "chunks", ["chunk_hash"])

    op.create_table(
        "scrape_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("sources.id"), nullable=False),
        sa.Column("status", scrape_job_status_enum, nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_scrape_jobs_source_id", "scrape_jobs", ["source_id"])


def downgrade() -> None:
    op.drop_table("scrape_jobs")
    op.drop_table("chunks")
    op.drop_table("sources")
    op.drop_table("admins")
    op.execute("DROP EXTENSION IF EXISTS vector")
