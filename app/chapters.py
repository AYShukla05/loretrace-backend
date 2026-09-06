"""Deterministic per-chunk tradition overrides for general-mythology volumes.

Most sources cover one tradition, so a chunk inherits its source's. A few
survey books do not: Bulfinch's *The Age of Fable* (Gutenberg #4925) is
tagged "Greek" for its Greco-Roman bulk, but has whole chapters on Norse
and Celtic myth and sub-sections on Egyptian, Persian, Hindu and Buddhist
belief. Left alone, those chapters would only ever retrieve and advertise
as "Greek".

`assign_chunk_traditions` splits such a volume at its own ALL-CAPS heading
lines and tags each chunk with the tradition it falls in, or None to mean
"inherit the source's". Heading-driven and exact-match only, no LLM and no
fuzzy guessing, same rule as app.works. A volume with no table here gets
all None (the single-tradition default).
"""

from app.works import gutenberg_ebook_id, label_chunks

# Ordered top to bottom of the volume's body. When a line equals a marker's
# literal, every following chunk takes that marker's tradition until the
# next marker; None returns the span to the source's own tradition. Markers
# are resolved in order, each strictly after the previous one's line. The
# "CHAPTER N" entries are anchors: they carry no tradition change of their
# own, but they push the search past the front-matter contents list so a
# sub-section literal ("EGYPTIAN DEITIES", "GLOSSARY", ...) that also
# appears there can only open a span at the body heading.
_MARKERS: dict[str, list[tuple[str, str | None]]] = {
    "4925": [
        ("CHAPTER XXXIV", None),  # anchor past the contents list
        ("EGYPTIAN DEITIES", "Egyptian"),  # mid-chapter
        ("ORACLE OF TROPHONIUS", None),  # back to Greek for the oracles
        ("CHAPTER XXXVII", None),  # anchor
        ("ZOROASTER", "Persian"),  # ch XXXVII opens on Zoroastrianism
        ("HINDU MYTHOLOGY", "Hindu"),
        ("BUDDHA", "Buddhist"),  # Buddha + the Grand Lama
        ("PRESTER JOHN", None),  # a medieval European legend, neither
        ("CHAPTER XXXVIII", "Norse"),  # Northern Mythology through ch XL
        ("CHAPTER XLI", "Celtic"),  # the Druids
        ("GLOSSARY", None),  # the index is overwhelmingly Greco-Roman
    ],
}


def _tradition_boundaries(
    text: str, markers: list[tuple[str, str | None]]
) -> list[tuple[int, str | None]]:
    """(char offset, tradition) pairs, offset-sorted, the first at (0, None).
    Each marker is matched to the first standalone line equal to its literal
    that starts after the previous matched marker; a literal that never
    appears is skipped without derailing the ones after it.
    """
    line_offsets: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_offsets.append((offset, line.strip()))
        offset += len(line)

    boundaries: list[tuple[int, str | None]] = [(0, None)]
    search_from = 0
    for literal, tradition in markers:
        hit = next(
            (off for off, stripped in line_offsets if off >= search_from and stripped == literal),
            None,
        )
        if hit is None:
            continue
        boundaries.append((hit, tradition))
        search_from = hit + 1
    return boundaries


def assign_chunk_traditions(text: str, chunk_texts: list[str], source_url: str) -> list[str | None]:
    """One tradition override (or None to inherit the source's) per chunk,
    in the given order. All None for a source with no table here.
    """
    markers = _MARKERS.get(gutenberg_ebook_id(source_url or "") or "")
    if markers is None:
        return [None] * len(chunk_texts)

    return label_chunks(text, chunk_texts, _tradition_boundaries(text, markers))
