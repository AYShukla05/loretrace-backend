import asyncio

from app.db.session import async_session
from app.worker.queue import run_once


async def run_worker(poll_interval: float = 5.0) -> None:
    while True:
        async with async_session() as db:
            claimed = await run_once(db)
        if not claimed:
            await asyncio.sleep(poll_interval)
