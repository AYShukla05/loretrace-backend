"""add source title

Revision ID: 504ebc630a24
Revises: 71f2a63e7ad6
Create Date: 2026-08-16 21:56:34.022840

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "504ebc630a24"
down_revision: str | None = "71f2a63e7ad6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("title", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("sources", "title")
