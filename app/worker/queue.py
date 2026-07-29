from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ScrapeJobStatus, SourceStatus
from app.models.scrape_job import ScrapeJob
from app.models.source import Source
from app.worker.pipeline import process_source


def naive_utcnow() -> datetime:
    """Postgres timestamp columns here are `timestamp without time zone`;
    handing asyncpg a tz-aware datetime raises `can't subtract offset-naive
    and offset-aware datetimes`. Compute in UTC, then drop the tzinfo."""
    return datetime.now(UTC).replace(tzinfo=None)


async def claim_next_source(db: AsyncSession) -> Source | None:
    result = await db.execute(
        select(Source)
        .where(Source.status == SourceStatus.PENDING)
        .order_by(Source.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    source = result.scalar_one_or_none()
    if source is None:
        return None

    source.status = SourceStatus.SCRAPING
    await db.commit()
    await db.refresh(source)
    return source


async def run_once(db: AsyncSession) -> bool:
    """Claim and process one pending source. Returns False if the queue is empty."""
    source = await claim_next_source(db)
    if source is None:
        return False

    job = ScrapeJob(
        source_id=source.id,
        status=ScrapeJobStatus.RUNNING,
        attempts=1,
        started_at=naive_utcnow(),
    )
    db.add(job)
    await db.commit()

    try:
        await process_source(source, db)
    except Exception as exc:  # job worker boundary: one bad source must not kill the loop
        source.status = SourceStatus.FAILED
        source.last_checked_at = naive_utcnow()
        job.status = ScrapeJobStatus.FAILED
        job.error_message = str(exc)
        job.finished_at = naive_utcnow()
        await db.commit()
        return True

    now = naive_utcnow()
    source.status = SourceStatus.COMPLETED
    source.last_scraped_at = now
    source.last_checked_at = now
    job.status = ScrapeJobStatus.COMPLETED
    job.finished_at = now
    await db.commit()
    return True
