from app.dedup import content_unchanged, diff_chunks, hash_text


def test_hash_text_is_deterministic():
    assert hash_text("same text") == hash_text("same text")


def test_hash_text_differs_for_different_input():
    assert hash_text("text one") != hash_text("text two")


def test_hash_text_ignores_surrounding_whitespace():
    assert hash_text("some text") == hash_text("  some text\n")


def test_content_unchanged_true_when_hash_matches():
    previous = hash_text("the myth of Prometheus")
    assert content_unchanged(previous, "the myth of Prometheus") is True


def test_content_unchanged_false_when_hash_differs():
    previous = hash_text("the myth of Prometheus")
    assert content_unchanged(previous, "a different retelling") is False


def test_content_unchanged_false_when_no_previous_hash():
    assert content_unchanged(None, "first time seeing this text") is False


def test_diff_chunks_classifies_new_stale_and_unchanged():
    kept = hash_text("chunk that survives unedited")
    removed = hash_text("chunk that got cut in a later revision")
    added = hash_text("chunk that only appears in the new revision")

    diff = diff_chunks(
        existing_hashes=[kept, removed],
        new_chunk_texts=[
            "chunk that survives unedited",
            "chunk that only appears in the new revision",
        ],
    )

    assert diff.new == {added}
    assert diff.stale == {removed}
    assert diff.unchanged == {kept}


def test_diff_chunks_all_new_when_no_existing_hashes():
    diff = diff_chunks(existing_hashes=[], new_chunk_texts=["a fresh chunk", "another fresh chunk"])

    assert len(diff.new) == 2
    assert diff.stale == frozenset()
    assert diff.unchanged == frozenset()


def test_diff_chunks_all_stale_when_source_no_longer_yields_chunks():
    gone = hash_text("chunk from a source that's now empty")
    diff = diff_chunks(existing_hashes=[gone], new_chunk_texts=[])

    assert diff.new == frozenset()
    assert diff.stale == {gone}
    assert diff.unchanged == frozenset()
