"""resize chunk embedding dimension

Revision ID: 75ae6cdedde6
Revises: 7ad0cd812952
Create Date: 2026-07-31 22:41:17.203991

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "75ae6cdedde6"
down_revision: str | None = "7ad0cd812952"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# sentence-transformers/all-MiniLM-L6-v2 replaces text-embedding-3-small as the
# embedding model, see app/embedding.py and LoreTrace_AI_Layer_Decision.md
NEW_EMBEDDING_DIM = 384


def upgrade() -> None:
    # No embeddings have been computed yet (chunking/embedding service is new),
    # so there's no existing vector data to migrate, just a dimension to fix.
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(NEW_EMBEDDING_DIM), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "embedding")
    op.add_column("chunks", sa.Column("embedding", Vector(1536), nullable=True))
