import asyncio
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import embed_texts
from app.models.chunk import Chunk
from app.models.source import Source

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
    tradition: str
    chunk_text: str
    distance: float


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


async def retrieve_chunks(
    db: AsyncSession,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    tradition: str | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k closest active chunks by cosine
    distance, restricted to RELEVANCE_THRESHOLD. An empty result means
    nothing in the corpus is relevant enough to answer from, the caller
    should refuse rather than invoke the LLM.
    """
    (query_embedding,) = await asyncio.to_thread(embed_texts, [query])
    stmt = _build_query(query_embedding, top_k, tradition)

    rows = await db.execute(stmt)
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            source_id=source.id,
            source_url=source.url,
            tradition=source.tradition,
            chunk_text=chunk.chunk_text,
            distance=distance,
        )
        for chunk, source, distance in rows.all()
    ]
