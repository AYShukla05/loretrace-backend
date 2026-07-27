from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.models.enums import SourceStatus, SourceType

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType, native_enum=False))
    tradition: Mapped[str] = mapped_column(String(50), index=True)
    status: Mapped[SourceStatus] = mapped_column(
        Enum(SourceStatus, native_enum=False), default=SourceStatus.PENDING
    )
    content_hash: Mapped[str | None] = mapped_column(String(64))
    etag: Mapped[str | None] = mapped_column(String(255))
    last_modified: Mapped[str | None] = mapped_column(String(255))
    last_scraped_at: Mapped[datetime | None]
    last_checked_at: Mapped[datetime | None]
    added_by: Mapped[int] = mapped_column(ForeignKey("admins.id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="source")
