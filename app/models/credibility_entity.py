from datetime import datetime
from typing import Any

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base
from app.models.enums import CredibilityEntityType, SuggestionStatus, pg_enum


class CredibilityEntity(Base):
    """Cached author/institution lookup for the credibility-suggestion
    mechanism, see LoreTrace_Credibility_Suggestion_Design.md. One row per
    real-world entity, reused across every source that cites them.
    """

    __tablename__ = "credibility_entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[CredibilityEntityType] = mapped_column(pg_enum(CredibilityEntityType))
    display_name: Mapped[str] = mapped_column(String(255))
    normalized_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    # Shape deliberately loose (JSON, not columns): fact availability varies
    # wildly by entity. See design doc section 4.3 for the expected keys.
    facts: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    fact_provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    suggested_values: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    suggestion_status: Mapped[SuggestionStatus] = mapped_column(
        pg_enum(SuggestionStatus), default=SuggestionStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
