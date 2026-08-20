from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.db.session import get_db
from app.models.admin import Admin
from app.models.chunk import Chunk
from app.models.enums import ScrapeJobStatus, SourceStatus
from app.models.scrape_job import ScrapeJob
from app.models.source import Source
from app.schemas.source import SourceCreate, SourceRead, SourceUpdate

router = APIRouter(prefix="/admin/sources", tags=["admin"])


def _to_read(source: Source, chunk_count: int) -> SourceRead:
    return SourceRead(
        id=source.id,
        url=source.url,
        title=source.title,
        source_type=source.source_type,
        tradition=source.tradition,
        status=source.status,
        era=source.era,
        author_position=source.author_position,
        text_role=source.text_role,
        known_bias_flags=source.known_bias_flags,
        historiographical_method=source.historiographical_method,
        author_origin=source.author_origin,
        author_epistemic_basis=source.author_epistemic_basis,
        last_scraped_at=source.last_scraped_at,
        last_checked_at=source.last_checked_at,
        created_at=source.created_at,
        chunk_count=chunk_count,
    )


async def _chunk_count(db: AsyncSession, source_id: int) -> int:
    count = await db.scalar(
        select(func.count(Chunk.id)).where(Chunk.source_id == source_id, Chunk.is_active.is_(True))
    )
    return count or 0


@router.post("", response_model=SourceRead, status_code=status.HTTP_201_CREATED)
async def create_source(
    payload: SourceCreate,
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> SourceRead:
    url = str(payload.url)
    existing = await db.scalar(select(Source).where(Source.url == url))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Source URL already registered")

    source = Source(
        url=url,
        title=payload.title,
        source_type=payload.source_type,
        tradition=payload.tradition,
        status=SourceStatus.PENDING,
        era=payload.era,
        author_position=payload.author_position,
        text_role=payload.text_role,
        known_bias_flags=payload.known_bias_flags,
        historiographical_method=payload.historiographical_method,
        author_origin=payload.author_origin,
        author_epistemic_basis=payload.author_epistemic_basis,
        added_by=admin.id,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return _to_read(source, chunk_count=0)


@router.get("", response_model=list[SourceRead])
async def list_sources(
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> list[SourceRead]:
    chunk_counts = (
        select(Chunk.source_id, func.count(Chunk.id).label("chunk_count"))
        .where(Chunk.is_active.is_(True))
        .group_by(Chunk.source_id)
        .subquery()
    )
    rows = await db.execute(
        select(Source, func.coalesce(chunk_counts.c.chunk_count, 0))
        .outerjoin(chunk_counts, chunk_counts.c.source_id == Source.id)
        .order_by(Source.created_at.desc())
    )
    return [_to_read(source, chunk_count) for source, chunk_count in rows.all()]


@router.get("/traditions", response_model=list[str])
async def admin_traditions(
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> list[str]:
    """Every distinct tradition value on any source, regardless of chunk
    state. Unlike GET /chat/traditions (which only offers a tradition that
    would actually retrieve something for a public chat query), this backs
    an admin-side autocomplete, so a tradition an admin already used should
    show up here even before its source finishes scraping.
    """
    rows = await db.execute(
        select(Source.tradition)
        .where(Source.tradition.is_not(None))
        .distinct()
        .order_by(Source.tradition)
    )
    return [row[0] for row in rows.all()]


@router.patch("/{source_id}", response_model=SourceRead)
async def update_source(
    source_id: int,
    payload: SourceUpdate,
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> SourceRead:
    source = await db.get(Source, source_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Source not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(source, field, value)

    db.add(source)
    await db.commit()
    await db.refresh(source)
    return _to_read(source, await _chunk_count(db, source.id))


@router.post("/{source_id}/rescrape", response_model=SourceRead)
async def rescrape_source(
    source_id: int,
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> SourceRead:
    source = await db.get(Source, source_id)
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Source not found")

    source.status = SourceStatus.PENDING
    db.add(ScrapeJob(source_id=source.id, status=ScrapeJobStatus.PENDING))
    await db.commit()
    await db.refresh(source)
    return _to_read(source, await _chunk_count(db, source.id))
