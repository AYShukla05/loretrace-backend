"""Deterministic sub-work labelling for multi-work source volumes.

A few public-domain sources bundle many distinct works into one file:
Gutenberg #348 is Hesiod's *Theogony* and *Works and Days*, 33 Homeric
Hymns, the Epic Cycle fragments and more; #27458 is two Aeschylus plays.
Every chunk from such a volume otherwise cites by the same container
title, so a disagreement between two of its works (the *Theogony* has
Aphrodite foam-born, the *Homeric Hymn to Aphrodite* calls her a
daughter of Zeus) reads as one source contradicting itself.

`assign_work_titles` splits a volume at its own section headings and
tags each chunk with the work it falls in. Heading-driven and
exact-match only, no LLM and no fuzzy guessing, consistent with the
project's extractive-classification rule. A volume with no table here
is returned entirely unlabelled (every chunk None), which is the
single-work default.
"""

import re
from bisect import bisect_right

_FRONT_MATTER_TITLE = "Introduction"

# A heading line in this corpus may carry a trailing endnote number
# ("THE CATALOGUES OF WOMEN AND EOIAE1701", "I. TO DIONYSUS 2501").
_TRAILING_ENDNOTE = re.compile(r"\s*\d+\s*$")
_HYMN_HEADING = re.compile(r"^([IVXL]{1,6})\.\s+TO\s+(.+)$")
_SMART_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _normalize_heading(line: str) -> str:
    return _TRAILING_ENDNOTE.sub("", line.strip().translate(_SMART_QUOTES))


_EBOOK_ID_IN_URL = re.compile(r"/(?:ebooks|files|cache/epub)/(\d+)")

# Small words that stay lowercased inside a hymn's dedication when they
# aren't the first word ("Homeric Hymn 23 to the Son of Cronos").
_LOWERCASE_WORDS = {"the", "of", "to", "and", "a", "an", "in"}


class _Marker:
    __slots__ = ("literal", "work_title", "opens_hymns")

    def __init__(self, literal: str, work_title: str, opens_hymns: bool = False):
        self.literal = literal
        self.work_title = work_title
        self.opens_hymns = opens_hymns


# Ordered top to bottom. When a volume line equals a marker's literal
# (after trailing-endnote strip), every following chunk takes that
# marker's work_title until the next marker. A marker with opens_hymns
# also turns on per-hymn heading detection until the following marker.
_TABLES: dict[str, list[_Marker]] = {
    "348": [
        _Marker("HESIOD'S WORKS AND DAYS", "Works and Days"),
        _Marker("THE DIVINATION BY BIRDS", "Hesiodic Fragments"),
        _Marker("THE THEOGONY", "Theogony"),
        _Marker("THE CATALOGUES OF WOMEN AND EOIAE", "Catalogues of Women and Eoiae"),
        _Marker("THE SHIELD OF HERACLES", "The Shield of Heracles"),
        _Marker("THE MARRIAGE OF CEYX", "Hesiodic Fragments"),
        _Marker("THE HOMERIC HYMNS", "The Homeric Hymns", opens_hymns=True),
        _Marker("FRAGMENTS OF THE EPIC CYCLE", "Fragments of the Epic Cycle"),
        _Marker("HOMERICA", "Homerica"),
        _Marker(
            "OF THE ORIGIN OF HOMER AND HESIOD, AND OF THEIR CONTEST",
            "The Contest of Homer and Hesiod",
        ),
        _Marker("ENDNOTES", "Endnotes"),
    ],
    "27458": [
        _Marker("PROMETHEUS CHAINED.", "Prometheus Bound"),
        _Marker("THE SEVEN AGAINST THEBES.", "Seven Against Thebes"),
    ],
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50}


def _roman_to_int(roman: str) -> int:
    total = 0
    prev = 0
    for char in reversed(roman):
        value = _ROMAN_VALUES[char]
        total += -value if value < prev else value
        prev = value
    return total


def _titlecase_dedication(name: str) -> str:
    out = []
    for word in name.strip().split():
        lower = word.lower()
        if lower.strip(",") in _LOWERCASE_WORDS:
            out.append(lower)
        else:
            out.append(word[:1].upper() + word[1:].lower())
    return " ".join(out)


def _hymn_title(match: re.Match) -> str:
    number = _roman_to_int(match.group(1))
    return f"Homeric Hymn {number} to {_titlecase_dedication(match.group(2))}"


def gutenberg_ebook_id(url: str) -> str | None:
    match = _EBOOK_ID_IN_URL.search(url)
    return match.group(1) if match else None


def _boundaries(text: str, markers: list[_Marker]) -> list[tuple[int, str]]:
    """(char offset, work_title) pairs, offset-sorted, one per heading the
    volume actually contains. Always starts with the front-matter entry.

    A volume repeats its section headings in a contents block near the top.
    The body heading is always the *last* occurrence of each, so that wins;
    per-hymn headings are only taken from between the body "THE HOMERIC
    HYMNS" line and the section that follows the hymns.
    """
    hymns_marker = next((i for i, m in enumerate(markers) if m.opens_hymns), None)

    last_literal: dict[int, int] = {}
    hymn_hits: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = _normalize_heading(line)
        for i, marker in enumerate(markers):
            if stripped == marker.literal:
                last_literal[i] = offset
        hymn = _HYMN_HEADING.match(stripped)
        if hymn is not None:
            hymn_hits.append((offset, _hymn_title(hymn)))
        offset += len(line)

    result: list[tuple[int, str]] = [(0, _FRONT_MATTER_TITLE)]
    for i, marker in enumerate(markers):
        if i in last_literal:
            result.append((last_literal[i], marker.work_title))

    if hymns_marker is not None and hymns_marker in last_literal:
        hymns_start = last_literal[hymns_marker]
        after = range(hymns_marker + 1, len(markers))
        later = [last_literal[i] for i in after if i in last_literal]
        hymns_end = min(later) if later else len(text)
        result.extend((off, title) for off, title in hymn_hits if hymns_start < off < hymns_end)

    result.sort()
    return result


def _chunk_offset(chunk_text: str, text: str, search_from: int) -> int:
    """Where chunk_text starts in the original text. Chunks collapse
    whitespace, so the lookup is a whitespace-flexible match on the
    chunk's opening words.
    """
    words = chunk_text.split()[:8]
    if not words:
        return -1
    pattern = re.compile(r"\s+".join(re.escape(word) for word in words))
    match = pattern.search(text, search_from)
    return match.start() if match else -1


def label_chunks(
    text: str, chunk_texts: list[str], boundaries: list[tuple[int, str | None]]
) -> list[str | None]:
    """Give each chunk the label of the last boundary at or before where its
    text starts in `text`. `boundaries` is (char offset, label) pairs,
    offset-sorted, the first at offset 0. A chunk whose opening words can't
    be located keeps the previous chunk's label.

    Shared by assign_work_titles here and app.chapters.assign_chunk_traditions,
    which differ only in what the boundaries mean.
    """
    offsets = [offset for offset, _ in boundaries]
    labels = [label for _, label in boundaries]

    result: list[str | None] = []
    cursor = 0
    last = labels[0]
    for chunk_text in chunk_texts:
        position = _chunk_offset(chunk_text, text, cursor)
        if position == -1:
            result.append(last)
            continue
        cursor = position
        last = labels[bisect_right(offsets, position) - 1]
        result.append(last)
    return result


def assign_work_titles(text: str, chunk_texts: list[str], source_url: str) -> list[str | None]:
    """One work_title (or None) per chunk, in the given order. None for a
    source whose ebook id has no table here, i.e. a single-work source.
    """
    ebook_id = gutenberg_ebook_id(source_url or "")
    markers = _TABLES.get(ebook_id or "")
    if markers is None:
        return [None] * len(chunk_texts)

    return label_chunks(text, chunk_texts, _boundaries(text, markers))
