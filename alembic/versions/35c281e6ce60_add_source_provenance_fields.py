"""add source provenance fields

Revision ID: 35c281e6ce60
Revises: 75ae6cdedde6
Create Date: 2026-07-31 22:58:02.698053

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "35c281e6ce60"
down_revision: str | None = "75ae6cdedde6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

era_enum = sa.Enum(
    "pre_colonial",
    "colonial_era",
    "post_independence",
    "contemporary",
    name="era",
    native_enum=False,
)
author_position_enum = sa.Enum(
    "indigenous_primary_text",
    "indigenous_scholar",
    "colonial_administrator",
    "missionary",
    "western_academic",
    "unknown_compiler",
    name="authorposition",
    native_enum=False,
)
text_role_enum = sa.Enum(
    "primary_translation",
    "secondary_commentary",
    "tertiary_summary",
    name="textrole",
    native_enum=False,
)


def upgrade() -> None:
    op.add_column("sources", sa.Column("era", era_enum, nullable=True))
    op.add_column("sources", sa.Column("author_position", author_position_enum, nullable=True))
    op.add_column("sources", sa.Column("text_role", text_role_enum, nullable=True))
    op.add_column("sources", sa.Column("known_bias_flags", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("sources", "known_bias_flags")
    op.drop_column("sources", "text_role")
    op.drop_column("sources", "author_position")
    op.drop_column("sources", "era")
