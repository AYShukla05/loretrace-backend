"""add super admin role

Revision ID: b8f44c911eb1
Revises: 11616879e8c6
Create Date: 2026-08-08 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8f44c911eb1"
down_revision: str | None = "11616879e8c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("admins", "is_admin", new_column_name="is_active")
    op.add_column(
        "admins",
        sa.Column("is_super_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("admins", "is_super_admin", server_default=None)


def downgrade() -> None:
    op.drop_column("admins", "is_super_admin")
    op.alter_column("admins", "is_active", new_column_name="is_admin")
