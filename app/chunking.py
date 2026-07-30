import re

DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(paragraph)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current and current_len + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into word-count-bounded chunks, preferring paragraph boundaries.

    Paragraphs are packed together up to chunk_size words. A paragraph longer
    than chunk_size on its own is split on sentence boundaries instead, with
    overlap carried between the resulting pieces so context isn't lost at the
    cut.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
        current = []
        current_len = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > chunk_size:
            flush()
            chunks.extend(_split_long_paragraph(paragraph, chunk_size, overlap))
            continue
        if current_len + len(words) > chunk_size:
            flush()
        current.extend(words)
        current_len += len(words)

    flush()
    return chunks
