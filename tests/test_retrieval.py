from sqlalchemy.dialects import postgresql

from app.models.enums import AuthorPosition
from app.retrieval import (
    MAX_CHUNKS_PER_SOURCE,
    MAX_FLOOR_SOURCES,
    RELEVANCE_THRESHOLD,
    SOURCE_FLOOR_DISTANCE,
    RetrievedChunk,
    _build_candidate_query,
    _build_corpus_query,
    _build_traditions_query,
    _select_with_source_floor,
    _sort_by_provenance,
)


def make_chunk(chunk_id: int, author_position: AuthorPosition | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id=chunk_id,
        source_url=f"https://example.com/{chunk_id}",
        tradition="test",
        chunk_text="text",
        distance=0.1 * chunk_id,
        author_position=author_position,
    )


def pool_chunk(chunk_id: int, source_id: int, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_url=f"https://example.com/{source_id}",
        tradition="test",
        chunk_text="text",
        distance=distance,
    )


def test_build_candidate_query_filters_active_chunks_within_floor_distance():
    stmt = _build_candidate_query([0.1] * 384, tradition=None)
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "chunks.is_active IS true" in compiled
    assert f"<= {SOURCE_FLOOR_DISTANCE}" in compiled
    assert "ORDER BY" in compiled


def test_build_candidate_query_filters_by_tradition_when_given():
    stmt = _build_candidate_query([0.1] * 384, tradition="greek")
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "sources.tradition" in compiled
    assert "greek" in compiled


def test_build_candidate_query_omits_tradition_filter_when_none():
    stmt = _build_candidate_query([0.1] * 384, tradition=None)
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "sources.tradition =" not in compiled


def test_build_candidate_query_joins_chunks_to_their_source():
    stmt = _build_candidate_query([0.1] * 384, tradition=None)
    compiled = str(stmt.compile(dialect=postgresql.dialect()))

    assert "JOIN sources ON chunks.source_id = sources.id" in compiled


def test_select_returns_empty_when_nothing_within_relevance_threshold():
    pool = [pool_chunk(1, source_id=1, distance=RELEVANCE_THRESHOLD + 0.05)]

    assert _select_with_source_floor(pool, top_k=5) == []


def test_select_caps_chunks_from_a_single_source():
    pool = [pool_chunk(i, source_id=1, distance=0.1 + i * 0.01) for i in range(6)]

    result = _select_with_source_floor(pool, top_k=5)

    assert len(result) == MAX_CHUNKS_PER_SOURCE


def test_select_prefers_other_sources_once_one_is_capped():
    dominant = [pool_chunk(i, source_id=1, distance=0.10 + i * 0.001) for i in range(5)]
    other = [pool_chunk(100, source_id=2, distance=0.30)]

    result = _select_with_source_floor(dominant + other, top_k=5)
    by_source = {c.source_id for c in result}

    assert by_source == {1, 2}
    assert sum(c.source_id == 1 for c in result) == MAX_CHUNKS_PER_SOURCE


def test_select_floors_in_an_unrepresented_source_from_the_looser_band():
    primary = [pool_chunk(1, source_id=1, distance=0.20)]
    # source 2 has nothing within RELEVANCE_THRESHOLD, only the looser band
    far = [pool_chunk(2, source_id=2, distance=RELEVANCE_THRESHOLD + 0.1)]

    result = _select_with_source_floor(primary + far, top_k=5)

    assert {c.source_id for c in result} == {1, 2}


def test_select_floor_is_capped_and_takes_each_source_best_chunk():
    primary = [pool_chunk(1, source_id=1, distance=0.20)]
    far = [
        pool_chunk(10, source_id=2, distance=0.40),
        pool_chunk(11, source_id=2, distance=0.45),  # worse chunk, same source
        pool_chunk(20, source_id=3, distance=0.41),
        pool_chunk(30, source_id=4, distance=0.42),
        pool_chunk(40, source_id=5, distance=0.43),  # beyond MAX_FLOOR_SOURCES
    ]

    result = _select_with_source_floor(primary + far, top_k=5)
    floored = [c for c in result if c.source_id != 1]

    assert len(floored) == MAX_FLOOR_SOURCES
    assert 11 not in {c.chunk_id for c in result}  # only source 2's best


def test_select_result_is_ordered_by_distance():
    pool = [
        pool_chunk(1, source_id=1, distance=0.10),
        pool_chunk(2, source_id=1, distance=0.20),
        pool_chunk(3, source_id=2, distance=RELEVANCE_THRESHOLD + 0.05),
    ]

    result = _select_with_source_floor(pool, top_k=5)

    assert [c.distance for c in result] == sorted(c.distance for c in result)


def test_sort_by_provenance_moves_indigenous_primary_text_first():
    western = make_chunk(1, AuthorPosition.WESTERN_ACADEMIC)
    indigenous = make_chunk(2, AuthorPosition.INDIGENOUS_PRIMARY_TEXT)

    result = _sort_by_provenance([western, indigenous])

    assert [chunk.chunk_id for chunk in result] == [2, 1]


def test_sort_by_provenance_moves_indigenous_scholar_before_unlabeled():
    unlabeled = make_chunk(1, None)
    scholar = make_chunk(2, AuthorPosition.INDIGENOUS_SCHOLAR)

    result = _sort_by_provenance([unlabeled, scholar])

    assert [chunk.chunk_id for chunk in result] == [2, 1]


def test_sort_by_provenance_is_stable_within_the_same_tier():
    first = make_chunk(1, AuthorPosition.MISSIONARY)
    second = make_chunk(2, AuthorPosition.WESTERN_ACADEMIC)

    result = _sort_by_provenance([first, second])

    assert [chunk.chunk_id for chunk in result] == [1, 2]


def test_sort_by_provenance_never_drops_chunks():
    chunks = [make_chunk(i, None) for i in range(1, 4)]

    result = _sort_by_provenance(chunks)

    assert len(result) == 3


def test_build_traditions_query_only_counts_active_chunks():
    stmt = _build_traditions_query()
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "chunks.is_active IS true" in compiled
    assert "sources.tradition IS NOT NULL" in compiled
    assert "DISTINCT" in compiled
    assert "ORDER BY" in compiled


def test_build_corpus_query_only_includes_sources_with_active_chunks():
    stmt = _build_corpus_query()
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "JOIN chunks ON chunks.source_id = sources.id" in compiled
    assert "chunks.is_active IS true" in compiled
    assert "sources.tradition IS NOT NULL" in compiled
    assert "DISTINCT" in compiled
    assert "ORDER BY sources.tradition, sources.title" in compiled
