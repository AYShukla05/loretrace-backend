from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.source import Source

# Must match app.embedding.EMBEDDING_DIM (the active embedding model's output size)
EMBEDDING_DIM = 384


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    # The specific work within a multi-work source (e.g. "Theogony" or
    # "Homeric Hymn 5 to Aphrodite" inside a single Gutenberg volume),
    # assigned deterministically from the text's own section headings during
    # scraping. Null for single-work sources, which cite by Source.title.
    work_title: Mapped[str | None] = mapped_column(String(255))
    # Overrides Source.tradition for a single source that spans more than one
    # (e.g. a general-mythology volume with distinct Norse and Egyptian
    # chapters). Null means inherit the source's tradition; retrieval reads
    # COALESCE(chunk.tradition, source.tradition).
    tradition: Mapped[str | None] = mapped_column(String(50))
    chunk_text: Mapped[str] = mapped_column(Text)
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    source: Mapped["Source"] = relationship(back_populates="chunks")
