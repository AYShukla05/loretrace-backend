from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.models.enums import (
    AuthorEpistemicBasis,
    AuthorOrigin,
    AuthorPosition,
    Era,
    HistoriographicalMethod,
    SourceStatus,
    SourceType,
    TextRole,
    pg_enum,
)

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    # Nullable: only url is required at creation, admins fill these in later.
    source_type: Mapped[SourceType | None] = mapped_column(pg_enum(SourceType))
    tradition: Mapped[str | None] = mapped_column(String(50), index=True)
    status: Mapped[SourceStatus] = mapped_column(
        pg_enum(SourceStatus), default=SourceStatus.PENDING
    )
    # Provenance, per LoreTrace_Bias_Mitigation_Plan.md Part 2. Admin-entered at
    # ingestion, never model-inferred. Nullable until the admin ingestion UI
    # exists to collect them; contradiction-flagging degrades to unattributed
    # presentation for sources missing them rather than failing.
    era: Mapped[Era | None] = mapped_column(pg_enum(Era))
    author_position: Mapped[AuthorPosition | None] = mapped_column(pg_enum(AuthorPosition))
    text_role: Mapped[TextRole | None] = mapped_column(pg_enum(TextRole))
    known_bias_flags: Mapped[str | None] = mapped_column(Text)
    # Independent of author_position, see
    # LoreTrace_Credibility_Suggestion_Design.md section 4.1. Populated either
    # by direct admin entry or, once built, admin-confirmed suggestions from
    # the credibility-suggestion mechanism below.
    historiographical_method: Mapped[HistoriographicalMethod | None] = mapped_column(
        pg_enum(HistoriographicalMethod)
    )
    author_origin: Mapped[AuthorOrigin | None] = mapped_column(pg_enum(AuthorOrigin))
    author_epistemic_basis: Mapped[AuthorEpistemicBasis | None] = mapped_column(
        pg_enum(AuthorEpistemicBasis)
    )
    # The source's primary author/publisher, for suggestion caching. A second
    # source by the same author reuses this row rather than triggering a new
    # lookup.
    credibility_entity_id: Mapped[int | None] = mapped_column(ForeignKey("credibility_entities.id"))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    etag: Mapped[str | None] = mapped_column(String(255))
    last_modified: Mapped[str | None] = mapped_column(String(255))
    last_scraped_at: Mapped[datetime | None]
    last_checked_at: Mapped[datetime | None]
    added_by: Mapped[int] = mapped_column(ForeignKey("admins.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="source")
