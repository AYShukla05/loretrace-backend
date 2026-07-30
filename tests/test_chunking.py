import pytest

from app.chunking import split_into_chunks


def test_short_text_returns_single_chunk():
    text = "Thor is a god of thunder.\n\nHe wields the hammer Mjolnir."
    chunks = split_into_chunks(text, chunk_size=50, overlap=10)
    assert chunks == ["Thor is a god of thunder. He wields the hammer Mjolnir."]


def test_packs_multiple_paragraphs_up_to_chunk_size():
    paragraphs = [f"Paragraph {i} has exactly six words here." for i in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = split_into_chunks(text, chunk_size=14, overlap=2)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 14


def test_never_splits_a_paragraph_that_fits_in_one_chunk():
    text = "One short paragraph.\n\nAnother short paragraph."
    chunks = split_into_chunks(text, chunk_size=3, overlap=1)

    assert "One short paragraph." in chunks
    assert "Another short paragraph." in chunks


def test_long_paragraph_is_split_on_sentence_boundaries():
    sentences = [f"This is sentence number {i}." for i in range(10)]
    paragraph = " ".join(sentences)
    chunks = split_into_chunks(paragraph, chunk_size=15, overlap=3)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 15


def test_overlap_carries_words_between_split_chunks_of_long_paragraph():
    sentences = [f"This is sentence number {i}." for i in range(10)]
    paragraph = " ".join(sentences)
    chunks = split_into_chunks(paragraph, chunk_size=15, overlap=3)

    first_tail = chunks[0].split()[-3:]
    second_head = chunks[1].split()[:3]
    assert first_tail == second_head


def test_blank_input_returns_no_chunks():
    assert split_into_chunks("   \n\n   ") == []


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        split_into_chunks("some text", chunk_size=0, overlap=0)


def test_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_into_chunks("some text", chunk_size=10, overlap=10)
