"""add per-chunk work_title and tradition override columns

Revision ID: 6ea799b5c12e
Revises: c764a15c993f
Create Date: 2026-09-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6ea799b5c12e"
down_revision: str | None = "c764a15c993f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("work_title", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("tradition", sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "tradition")
    op.drop_column("chunks", "work_title")
