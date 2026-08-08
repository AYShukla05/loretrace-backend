"""relax source required fields to url-only at creation

Revision ID: a10597d5e2f4
Revises: b8f44c911eb1
Create Date: 2026-08-08 21:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a10597d5e2f4"
down_revision: str | None = "b8f44c911eb1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("sources", "source_type", existing_type=sa.String(), nullable=True)
    op.alter_column("sources", "tradition", existing_type=sa.String(length=50), nullable=True)


def downgrade() -> None:
    op.alter_column("sources", "tradition", existing_type=sa.String(length=50), nullable=False)
    op.alter_column("sources", "source_type", existing_type=sa.String(), nullable=False)
