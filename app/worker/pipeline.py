from sqlalchemy.ext.asyncio import AsyncSession

from app.models.source import Source


async def process_source(source: Source, db: AsyncSession) -> None:
    raise NotImplementedError("scrape/dedup/chunk/embed pipeline not implemented yet")
