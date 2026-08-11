"""add credibility suggestion schema

Revision ID: 71f2a63e7ad6
Revises: a10597d5e2f4
Create Date: 2026-08-11 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "71f2a63e7ad6"
down_revision: str | None = "a10597d5e2f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

historiographical_method_enum = sa.Enum(
    "oral_tradition",
    "textual_critical",
    "colonial_comparative_mythology",
    "archaeological_correlation",
    "archaeoastronomical_dating",
    "genetic_anthropological",
    "modern_academic_consensus",
    "unspecified",
    name="historiographicalmethod",
    native_enum=False,
)
author_origin_enum = sa.Enum(
    "indigenous_born",
    "foreign_born",
    "unknown",
    name="authororigin",
    native_enum=False,
)
author_epistemic_basis_enum = sa.Enum(
    "lived_practice",
    "textual_study_only",
    "mixed",
    "unknown",
    name="authorepistemicbasis",
    native_enum=False,
)
credibility_entity_type_enum = sa.Enum(
    "author",
    "institution",
    name="credibilityentitytype",
    native_enum=False,
)
suggestion_status_enum = sa.Enum(
    "pending",
    "reviewed",
    name="suggestionstatus",
    native_enum=False,
)


def upgrade() -> None:
    op.create_table(
        "credibility_entities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("entity_type", credibility_entity_type_enum, nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("normalized_key", sa.String(length=255), nullable=False),
        sa.Column("facts", JSONB(), nullable=False),
        sa.Column("fact_provenance", JSONB(), nullable=False),
        sa.Column("suggested_values", JSONB(), nullable=False),
        sa.Column(
            "suggestion_status",
            suggestion_status_enum,
            nullable=False,
            server_default="pending",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_credibility_entities_normalized_key",
        "credibility_entities",
        ["normalized_key"],
        unique=True,
    )

    op.add_column(
        "sources",
        sa.Column("historiographical_method", historiographical_method_enum, nullable=True),
    )
    op.add_column("sources", sa.Column("author_origin", author_origin_enum, nullable=True))
    op.add_column(
        "sources", sa.Column("author_epistemic_basis", author_epistemic_basis_enum, nullable=True)
    )
    op.add_column("sources", sa.Column("credibility_entity_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_sources_credibility_entity_id",
        "sources",
        "credibility_entities",
        ["credibility_entity_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_sources_credibility_entity_id", "sources", type_="foreignkey")
    op.drop_column("sources", "credibility_entity_id")
    op.drop_column("sources", "author_epistemic_basis")
    op.drop_column("sources", "author_origin")
    op.drop_column("sources", "historiographical_method")

    op.drop_index("ix_credibility_entities_normalized_key", table_name="credibility_entities")
    op.drop_table("credibility_entities")
