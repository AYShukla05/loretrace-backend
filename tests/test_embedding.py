from app.embedding import EMBEDDING_DIM, QUERY_INSTRUCTION, embed_texts


def test_empty_input_returns_empty_list():
    assert embed_texts([]) == []


def test_embeds_each_text_at_the_configured_dimension():
    vectors = embed_texts(["Thor wields Mjolnir.", "Zeus rules from Olympus."])
    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIM for vector in vectors)


def test_similar_texts_are_closer_than_unrelated_ones():
    thor_a, thor_b, unrelated = embed_texts(
        [
            "Thor is the Norse god of thunder.",
            "Thor, god of thunder in Norse mythology.",
            "The recipe calls for two cups of flour.",
        ]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(thor_a, thor_b) > cosine(thor_a, unrelated)


def test_query_mode_embeds_differently_than_document_mode():
    text = "Who wields Mjolnir?"
    (query_vector,) = embed_texts([text], is_query=True)
    (document_vector,) = embed_texts([text])
    assert query_vector != document_vector


def test_query_mode_matches_manually_prefixed_document_mode():
    text = "Who wields Mjolnir?"
    (query_vector,) = embed_texts([text], is_query=True)
    (prefixed_vector,) = embed_texts([QUERY_INSTRUCTION + text])
    assert query_vector == prefixed_vector
