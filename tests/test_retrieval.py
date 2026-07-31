from sqlalchemy.dialects import postgresql

from app.retrieval import RELEVANCE_THRESHOLD, _build_query


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
