from datetime import datetime

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import ScrapeJobStatus, pg_enum


class ScrapeJob(Base):
    __tablename__ = "scrape_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    status: Mapped[ScrapeJobStatus] = mapped_column(
        pg_enum(ScrapeJobStatus), default=ScrapeJobStatus.PENDING
    )
    attempts: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
