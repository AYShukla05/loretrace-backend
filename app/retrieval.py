import asyncio
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding import embed_texts
from app.models.chunk import Chunk
from app.models.enums import AuthorPosition, Era, TextRole
from app.models.source import Source
from app.theonyms import expand_query

# A chunk's effective tradition: its own override if set, otherwise the
# source's. One general-mythology volume (e.g. Bulfinch) can carry distinct
# Norse, Egyptian and Hindu chapters even though the source row is tagged
# "Greek"; every tradition read (the filter, the traditions list, the corpus
# overview) goes through this so those chapters are found and advertised
# under their real pantheon rather than the container's.
_EFFECTIVE_TRADITION = func.coalesce(Chunk.tradition, Source.tradition)

# Bias Mitigation Plan Part 2: default ordering (not filtering) prefers
# indigenous primary sources when present. Everything else, including
# sources with no author_position yet, keeps its retrieval (distance) order.
_PROVENANCE_PRIORITY = {
    AuthorPosition.INDIGENOUS_PRIMARY_TEXT: 0,
    AuthorPosition.INDIGENOUS_SCHOLAR: 1,
}

# Cosine distance from pgvector's `<=>` operator: 0 is identical, 1 is
# orthogonal, 2 is opposite. Tuned against real production data
# (LoreTrace_RelevanceThreshold_Tuning_Results.md, 2026-08-22): recall
# against Gate 1's 12 ground-truth queries plateaus at 8/12 from 0.35
# upward, so this costs no recall versus the already-accepted 0.65
# baseline, while cutting a 12-probe leakage sweep's pass-through rate
# from 12/12 down to 3/12. The 3 that remain are all generic,
# no-tradition-named, thematically-adjacent queries (e.g. "flood myths
# across cultures") that sit close to real Norse content in embedding
# space for a structural reason, not a tuning gap — no single distance
# cutoff separates them without destroying recall. That residual is a
# documented, accepted limitation (see the results doc and
# LoreTrace_Quality_Gates.md Gate 2), not fully closed by this value.
RELEVANCE_THRESHOLD = 0.35
# 10, not 5: when two texts in a tradition disagree, the weaker-matching
# side can sit several ranks below the stronger one while still being a
# genuine hit under RELEVANCE_THRESHOLD (LoreTrace_Greek_Retrieval_
# Characterization.md, the Aphrodite parentage case — the Homeric Hymn's
# "daughter of Zeus" chunks rank 7-15 where Hesiod's foam-birth ranks 1).
# A larger k lets both sides reach the answer without loosening the
# threshold. Still bounded well under the Groq free-tier per-minute token
# cap: ~13 chunks of context (10 plus the per-source floor) at a few
# hundred tokens each.
DEFAULT_TOP_K = 10

# A source with nothing in the primary band (within RELEVANCE_THRESHOLD)
# may still contribute its single best chunk if within this looser
# distance, so a question one text answers head-on and another only
# touches in passing still surfaces both. Without it, whichever text
# matches the query's vocabulary can crowd the others out of top_k
# entirely and a real disagreement between sources never appears. See
# LoreTrace_Greek_Retrieval_Characterization.md.
SOURCE_FLOOR_DISTANCE = 0.5
# No single source may occupy more than this many of the primary-band
# slots, so one vocabulary-matched text cannot monopolise the answer.
MAX_CHUNKS_PER_SOURCE = 3
# At most this many otherwise-unrepresented sources get a floor chunk.
MAX_FLOOR_SOURCES = 3
_CANDIDATE_POOL_SIZE = 100


@dataclass(frozen=True)
class CorpusEntry:
    tradition: str
    title: str | None
    url: str


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
    # The specific work within a multi-work volume (Source.title is the
    # container). None for a single-work source; see app/works.py.
    work_title: str | None = None


def _build_candidate_query(query_embedding: list[float], tradition: str | None) -> Select:
    """The candidate pool for _select_with_source_floor: active chunks
    within the looser SOURCE_FLOOR_DISTANCE, closest first. Selection down
    to the primary RELEVANCE_THRESHOLD and top_k happens in Python so the
    per-source cap and the floor pass can see the whole pool.
    """
    distance = Chunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(Chunk, Source, distance.label("distance"))
        .join(Source, Chunk.source_id == Source.id)
        .where(Chunk.is_active.is_(True), distance <= SOURCE_FLOOR_DISTANCE)
        .order_by(distance)
        .limit(_CANDIDATE_POOL_SIZE)
    )
    if tradition is not None:
        stmt = stmt.where(_EFFECTIVE_TRADITION == tradition)
    return stmt


def _select_with_source_floor(candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """From a distance-ordered candidate pool, take up to top_k chunks
    within RELEVANCE_THRESHOLD with no more than MAX_CHUNKS_PER_SOURCE from
    any one source, then let up to MAX_FLOOR_SOURCES otherwise-unrepresented
    sources add their single best chunk from the looser band.

    An empty primary band means refuse: the floor never turns a query that
    nothing answers within RELEVANCE_THRESHOLD into one that gets an answer,
    so the leakage-refusal boundary is unchanged.
    """
    primary = [c for c in candidates if c.distance <= RELEVANCE_THRESHOLD]
    if not primary:
        return []

    selected: list[RetrievedChunk] = []
    per_source: Counter[int] = Counter()
    for chunk in primary:  # candidates arrive closest-first
        if len(selected) >= top_k:
            break
        if per_source[chunk.source_id] >= MAX_CHUNKS_PER_SOURCE:
            continue
        selected.append(chunk)
        per_source[chunk.source_id] += 1

    represented = {c.source_id for c in selected}
    floor_added = 0
    for chunk in candidates:  # first appearance of a source is its best chunk
        if floor_added >= MAX_FLOOR_SOURCES:
            break
        if chunk.source_id in represented:
            continue
        selected.append(chunk)
        represented.add(chunk.source_id)
        floor_added += 1

    selected.sort(key=lambda c: c.distance)
    return selected


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
    tradition = _EFFECTIVE_TRADITION.label("tradition")
    return (
        select(tradition)
        .select_from(Source)
        .join(Chunk, Chunk.source_id == Source.id)
        .where(tradition.is_not(None), Chunk.is_active.is_(True))
        .distinct()
        .order_by(tradition)
    )


async def list_traditions(db: AsyncSession) -> list[str]:
    """Distinct tradition values with at least one retrievable chunk, so the
    chat UI's filter never offers an option that would always refuse.
    """
    stmt = _build_traditions_query()
    rows = await db.execute(stmt)
    return [row[0] for row in rows.all()]


def _build_corpus_query() -> Select:
    tradition = _EFFECTIVE_TRADITION.label("tradition")
    return (
        select(tradition, Source.title, Source.url)
        .select_from(Source)
        .join(Chunk, Chunk.source_id == Source.id)
        .where(tradition.is_not(None), Chunk.is_active.is_(True))
        .distinct()
        .order_by(tradition, Source.title)
    )


async def list_corpus(db: AsyncSession) -> list[CorpusEntry]:
    """Every source with at least one retrievable chunk, as flat
    (tradition, title, url) rows ordered by tradition then title. The chat
    landing groups these so a first-time visitor sees exactly which
    traditions and texts the corpus can answer from, rather than guessing
    at coverage one question at a time. Same only-advertise-what-retrieves
    rule as list_traditions.
    """
    rows = await db.execute(_build_corpus_query())
    return [
        CorpusEntry(tradition=tradition, title=title, url=url)
        for tradition, title, url in rows.all()
    ]


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

    The query is first passed through app.theonyms.expand_query so a
    question phrased in one text's vocabulary (e.g. "Aphrodite", "Zeus")
    can still reach a sibling text that uses the other names ("Venus",
    "Jove"). No-op unless the tradition has a theonym table and a grouped
    name appears.

    Selection applies a per-source cap and a floor for unrepresented
    sources (see _select_with_source_floor) so one text cannot crowd the
    others out of the answer.
    """
    expanded = expand_query(query, tradition)
    (query_embedding,) = await asyncio.to_thread(embed_texts, [expanded], is_query=True)
    stmt = _build_candidate_query(query_embedding, tradition)

    rows = await db.execute(stmt)
    candidates = [
        RetrievedChunk(
            chunk_id=chunk.id,
            source_id=source.id,
            source_url=source.url,
            tradition=chunk.tradition or source.tradition,
            chunk_text=chunk.chunk_text,
            distance=distance,
            author_position=source.author_position,
            era=source.era,
            text_role=source.text_role,
            known_bias_flags=source.known_bias_flags,
            title=source.title,
            work_title=chunk.work_title,
        )
        for chunk, source, distance in rows.all()
    ]
    return _sort_by_provenance(_select_with_source_floor(candidates, top_k))
