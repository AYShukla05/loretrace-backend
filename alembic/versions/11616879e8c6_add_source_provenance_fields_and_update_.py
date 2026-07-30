"""add source provenance fields and update embedding dimension

Revision ID: 11616879e8c6
Revises: 7ad0cd812952
Create Date: 2026-07-31 22:58:02.698053

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "11616879e8c6"
down_revision: str | None = "7ad0cd812952"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# sentence-transformers/all-MiniLM-L6-v2 replaces text-embedding-3-small as the
# embedding model, see app/embedding.py and LoreTrace_AI_Layer_Decision.md
NEW_EMBEDDING_DIM = 384

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

    # No embeddings have been computed yet (chunking/embedding service is new),
    # so there's no existing vector data to migrate, just a dimension to fix.
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(NEW_EMBEDDING_DIM), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(1536), nullable=True))

    op.drop_column("sources", "known_bias_flags")
    op.drop_column("sources", "text_role")
    op.drop_column("sources", "author_position")
    op.drop_column("sources", "era")
