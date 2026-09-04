from app.chunking import split_into_chunks
from app.works import assign_work_titles, gutenberg_ebook_id

# Miniature stand-in for Gutenberg #348: a contents block that repeats the
# section headings, then the body. Only the body headings should become
# boundaries. Chunked with a small size in the tests so each section
# produces its own chunk.
HESIOD_VOLUME = """\
Produced by a volunteer.

Contents

HESIOD'S WORKS AND DAYS
THE THEOGONY
THE HOMERIC HYMNS
V. TO APHRODITE
FRAGMENTS OF THE EPIC CYCLE

PREFACE

This edition collects the Hesiodic corpus with an introduction on its
transmission and its date and its manuscripts.

HESIOD'S WORKS AND DAYS

Muses of Pieria who give glory through song, come hither now and tell
of Zeus your father and his fixed and steady will.

THE THEOGONY

And men call her Aphrodite, the foam-born goddess and rich-crowned
Cytherea, because she grew amid the sea foam near Cythera.

THE HOMERIC HYMNS

I. TO DIONYSUS

For some say that you were born at Dracanum, and some on windy Icarus,
O Heaven-born, Insewn, far from all men.

V. TO APHRODITE 2510

Thereupon Aphrodite the daughter of Zeus answered the hero Anchises
and spoke to him with soft and gentle words.

XXIII. TO THE SON OF CRONOS, MOST HIGH

I will sing of Zeus, chiefest among the gods and greatest, whose word
is sure and whose counsel is deep.

FRAGMENTS OF THE EPIC CYCLE

The War of the Titans. Eumelus of Corinth is said to have composed the
Titanomachy in several books of verse.
"""

AESCHYLUS_VOLUME = """\
INTRODUCTION.

Aeschylus was born at Eleusis in 525 B.C. and fought at Marathon
against the invading armies of Persia.

PROMETHEUS CHAINED.

STRENGTH. We are come to the far-bounding plain of earth, to the
Scythian track, to the untrodden solitude beyond all roads.

THE SEVEN AGAINST THEBES.

ETEOCLES. Citizens of Cadmus, he who guards the city's business must
speak the word that fits the hour and not shrink from it.
"""

_SMALL = dict(chunk_size=20, overlap=0)


def _labelled(volume: str, url: str) -> dict[str, str | None]:
    chunks = split_into_chunks(volume, **_SMALL)
    titles = assign_work_titles(volume, chunks, url)
    return {chunk: title for chunk, title in zip(chunks, titles, strict=True)}


def _title_of(labelled: dict[str, str | None], needle: str) -> str | None:
    return next(title for chunk, title in labelled.items() if needle in chunk)


def test_ebook_id_parsed_from_various_gutenberg_url_shapes():
    assert gutenberg_ebook_id("https://www.gutenberg.org/ebooks/348") == "348"
    assert gutenberg_ebook_id("https://www.gutenberg.org/cache/epub/348/pg348.txt") == "348"
    assert gutenberg_ebook_id("https://www.gutenberg.org/files/27458/27458-0.txt") == "27458"
    assert gutenberg_ebook_id("https://en.wikipedia.org/wiki/Greek_mythology") is None


def test_single_work_source_gets_no_work_titles():
    chunks = ["first chunk text here", "second chunk text here"]
    assert assign_work_titles("body", chunks, "https://www.gutenberg.org/ebooks/2199") == [
        None,
        None,
    ]


def test_chunk_count_matches_input_for_untabled_source():
    chunks = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    titles = assign_work_titles("body", chunks, "https://www.gutenberg.org/ebooks/9999")
    assert titles == [None, None]


def test_multi_work_volume_labels_each_chunk_by_its_section():
    labelled = _labelled(HESIOD_VOLUME, "https://www.gutenberg.org/ebooks/348")
    assert _title_of(labelled, "foam-born goddess") == "Theogony"
    assert _title_of(labelled, "answered the hero Anchises") == "Homeric Hymn 5 to Aphrodite"
    assert _title_of(labelled, "Eumelus of Corinth") == "Fragments of the Epic Cycle"


def test_front_matter_before_the_first_section_is_labelled_introduction():
    labelled = _labelled(HESIOD_VOLUME, "https://www.gutenberg.org/ebooks/348")
    assert _title_of(labelled, "introduction on its") == "Introduction"


def test_repeated_headings_in_a_contents_block_do_not_create_boundaries():
    labelled = _labelled(HESIOD_VOLUME, "https://www.gutenberg.org/ebooks/348")
    # Text between the contents block and the body's first heading (the
    # PREFACE) would be mislabelled as a later section if the contents
    # lines had produced boundaries.
    assert _title_of(labelled, "introduction on its") == "Introduction"
    assert _title_of(labelled, "Muses of Pieria") == "Works and Days"
    # every hymn heading is detected in the body, not just the contents list
    assert _title_of(labelled, "windy Icarus") == "Homeric Hymn 1 to Dionysus"
    assert None not in labelled.values()


def test_hymn_number_is_arabic_and_small_words_are_lowercased():
    labelled = _labelled(HESIOD_VOLUME, "https://www.gutenberg.org/ebooks/348")
    # roman -> arabic, trailing endnote number ignored, "the" lowercased
    assert _title_of(labelled, "chiefest among the gods") == (
        "Homeric Hymn 23 to the Son of Cronos, Most High"
    )


def test_aeschylus_volume_splits_into_two_plays():
    labelled = _labelled(AESCHYLUS_VOLUME, "https://www.gutenberg.org/ebooks/27458")
    assert _title_of(labelled, "born at Eleusis") == "Introduction"
    assert _title_of(labelled, "far-bounding plain") == "Prometheus Bound"
    assert _title_of(labelled, "Citizens of Cadmus") == "Seven Against Thebes"
