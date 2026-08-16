import asyncio
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import embed_texts
from app.models.chunk import Chunk
from app.models.enums import AuthorPosition, Era, TextRole
from app.models.source import Source

# Bias Mitigation Plan Part 2: default ordering (not filtering) prefers
# indigenous primary sources when present. Everything else, including
# sources with no author_position yet, keeps its retrieval (distance) order.
_PROVENANCE_PRIORITY = {
    AuthorPosition.INDIGENOUS_PRIMARY_TEXT: 0,
    AuthorPosition.INDIGENOUS_SCHOLAR: 1,
}

# Cosine distance from pgvector's `<=>` operator: 0 is identical, 1 is
# orthogonal, 2 is opposite. Provisional cutoff, not yet validated against
# Gate 2's leakage-check probes in LoreTrace_Quality_Gates.md — retune once
# that eval exists.
RELEVANCE_THRESHOLD = 0.65
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source_id: int
    source_url: str
    tradition: str | None
    chunk_text: str
    distance: float
    author_position: AuthorPosition | None = None
    era: Era | None = None
    text_role: TextRole | None = None
    known_bias_flags: str | None = None
    title: str | None = None


def _build_query(query_embedding: list[float], top_k: int, tradition: str | None) -> Select:
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Source, distance.label("distance"))
        .join(Source, Chunk.source_id == Source.id)
        .where(Chunk.is_active.is_(True), distance <= RELEVANCE_THRESHOLD)
        .order_by(distance)
        .limit(top_k)
    )
    if tradition is not None:
        stmt = stmt.where(Source.tradition == tradition)
    return stmt


def _provenance_rank(chunk: RetrievedChunk) -> int:
    return _PROVENANCE_PRIORITY.get(chunk.author_position, len(_PROVENANCE_PRIORITY))


def _sort_by_provenance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorders (never drops) retrieved chunks so indigenous primary sources
    come first, per LoreTrace_Bias_Mitigation_Plan.md Part 2. A stable sort,
    so chunks within the same provenance tier keep their relative distance
    order.
    """
    return sorted(chunks, key=_provenance_rank)


def _build_traditions_query() -> Select:
    return (
        select(Source.tradition)
        .join(Chunk, Chunk.source_id == Source.id)
        .where(Source.tradition.is_not(None), Chunk.is_active.is_(True))
        .distinct()
        .order_by(Source.tradition)
    )


async def list_traditions(db: AsyncSession) -> list[str]:
    """Distinct tradition values with at least one retrievable chunk, so the
    chat UI's filter never offers an option that would always refuse.
    """
    stmt = _build_traditions_query()
    rows = await db.execute(stmt)
    return [row[0] for row in rows.all()]


async def retrieve_chunks(
    db: AsyncSession,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    tradition: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k closest active chunks by cosine
    distance, restricted to RELEVANCE_THRESHOLD, reordered to prefer
    indigenous primary sources. An empty result means nothing in the corpus
    is relevant enough to answer from, the caller should refuse rather than
    invoke the LLM.
    """
    (query_embedding,) = await asyncio.to_thread(embed_texts, [query])
    stmt = _build_query(query_embedding, top_k, tradition)

    rows = await db.execute(stmt)
    chunks = [
        RetrievedChunk(
            chunk_id=chunk.id,
            source_id=source.id,
            source_url=source.url,
            tradition=source.tradition,
            chunk_text=chunk.chunk_text,
            distance=distance,
            author_position=source.author_position,
            era=source.era,
            text_role=source.text_role,
            known_bias_flags=source.known_bias_flags,
            title=source.title,
        )
        for chunk, source, distance in rows.all()
    ]
    return _sort_by_provenance(chunks)
