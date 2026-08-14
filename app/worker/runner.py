import asyncio
import logging

from sqlalchemy import update

from app.core.config import settings
from app.db.session import async_session, engine
from app.models.enums import ScrapeJobStatus, SourceStatus
from app.models.scrape_job import ScrapeJob
from app.models.source import Source
from app.worker.queue import claim_next_source, naive_utcnow, run_claimed_job

logger = logging.getLogger(__name__)

# Bounds a real, previously-reproduced hang: a stale pooled Neon/PgBouncer
# connection can make db.commit() block forever with no error (see
# CLAUDE.md's 2026-08-15 session notes). Two narrower fixes were tried and
# confirmed not to work: an asyncpg-level command_timeout (doesn't bound
# commit() itself) and asyncio.wait_for around a single db.commit() (asyncio's
# wait_for awaits the cancelled task again to observe the CancelledError
# before raising TimeoutError, and that await never returns if the
# cancellation can't land inside SQLAlchemy's greenlet bridge — so wait_for
# itself hangs too). asyncio.wait(..., timeout=...), used below, never awaits
# the pending task at all, so it reliably returns on schedule regardless of
# whether the wedged task can ever be cancelled. The threshold itself lives in
# settings (see its own comment there), not here.
JOB_WATCHDOG_TIMEOUT = settings.worker_job_watchdog_seconds


async def run_worker(poll_interval: float = 5.0) -> None:
    while True:
        claimed = await _run_one_watched()
        if not claimed:
            await asyncio.sleep(poll_interval)


async def _run_one_watched(watchdog_timeout: float = JOB_WATCHDOG_TIMEOUT) -> bool:
    async with async_session() as claim_db:
        source = await claim_next_source(claim_db)
    if source is None:
        return False

    task = asyncio.ensure_future(_process_claimed(source))
    _done, pending = await asyncio.wait({task}, timeout=watchdog_timeout)
    if task in pending:
        logger.error(
            "Worker job for source %s exceeded %.0fs, likely a stale pooled "
            "connection; abandoning it and marking the source failed via a "
            "fresh connection",
            source.id,
            watchdog_timeout,
        )
        # Best-effort only, fire-and-forget: never awaited, so it can't wedge
        # this loop even if the cancellation never actually lands.
        task.cancel()
        # Discards idle pooled connections so the next checkout (including
        # the recovery write below) can't reuse whatever's wedged.
        await engine.dispose()
        message = f"worker watchdog timeout after {watchdog_timeout:.0f}s"
        await _mark_source_failed(source.id, message)
    return True


async def _process_claimed(source: Source) -> None:
    async with async_session() as db:
        await run_claimed_job(source, db)


async def _mark_source_failed(source_id: int, error_message: str) -> None:
    now = naive_utcnow()
    async with async_session() as db:
        await db.execute(
            update(Source)
            .where(Source.id == source_id)
            .values(status=SourceStatus.FAILED, last_checked_at=now)
        )
        db.add(
            ScrapeJob(
                source_id=source_id,
                status=ScrapeJobStatus.FAILED,
                attempts=1,
                started_at=now,
                finished_at=now,
                error_message=error_message,
            )
        )
        await db.commit()
