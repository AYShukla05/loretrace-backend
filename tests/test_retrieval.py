from sqlalchemy.dialects import postgresql

from app.models.enums import AuthorPosition
from app.retrieval import RELEVANCE_THRESHOLD, RetrievedChunk, _build_query, _sort_by_provenance


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


def test_build_query_filters_active_chunks_within_relevance_threshold():
    stmt = _build_query([0.1] * 384, top_k=5, tradition=None)
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "chunks.is_active IS true" in compiled
    assert f"<= {RELEVANCE_THRESHOLD}" in compiled
    assert "ORDER BY" in compiled
    assert "LIMIT 5" in compiled


def test_build_query_filters_by_tradition_when_given():
    stmt = _build_query([0.1] * 384, top_k=5, tradition="greek")
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "sources.tradition" in compiled
    assert "greek" in compiled


def test_build_query_omits_tradition_filter_when_none():
    stmt = _build_query([0.1] * 384, top_k=5, tradition=None)
    compiled = str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )

    assert "sources.tradition =" not in compiled


def test_build_query_joins_chunks_to_their_source():
    stmt = _build_query([0.1] * 384, top_k=5, tradition=None)
    compiled = str(stmt.compile(dialect=postgresql.dialect()))

    assert "JOIN sources ON chunks.source_id = sources.id" in compiled


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
