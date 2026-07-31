from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ScrapeJobStatus, SourceStatus
from app.models.scrape_job import ScrapeJob
from app.models.source import Source
from app.worker.pipeline import process_source


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
        started_at=datetime.now(UTC),
    )
    db.add(job)
    await db.commit()

    try:
        scraped = await process_source(source, db)
    except Exception as exc:  # job worker boundary: one bad source must not kill the loop
        source.status = SourceStatus.FAILED
        source.last_checked_at = datetime.now(UTC)
        job.status = ScrapeJobStatus.FAILED
        job.error_message = str(exc)
        job.finished_at = datetime.now(UTC)
        await db.commit()
        return True

    now = datetime.now(UTC)
    source.status = SourceStatus.COMPLETED
    if scraped:
        source.last_scraped_at = now
    source.last_checked_at = now
    job.status = ScrapeJobStatus.COMPLETED
    job.finished_at = now
    await db.commit()
    return True
