from app.chapters import assign_chunk_traditions
from app.chunking import split_into_chunks

# Miniature stand-in for Gutenberg #4925 (Bulfinch, The Age of Fable): a
# front-matter contents list that also names GLOSSARY, then a body whose
# ALL-CAPS heading lines drive the tradition overrides. Only the body
# headings should move the boundary; the contents GLOSSARY must not.
BULFINCH_VOLUME = """\
CONTENTS

PYTHAGORAS
EGYPTIAN DEITIES
ORACLES
EASTERN MYTHOLOGY
NORTHERN MYTHOLOGY
THE DRUIDS
GLOSSARY

CHAPTER II

PROMETHEUS AND PANDORA

Prometheus was one of the Titans, and he took some of this earth and
mixed it with water and made man in the image of the gods.

CHAPTER XXXIV

PYTHAGORAS

Pythagoras taught that numbers are the principle of all things and the
soul is immortal and passes into other bodies after death.

EGYPTIAN DEITIES

The Egyptians worshipped Osiris and Isis and their son Horus, and held
the ox Apis at Memphis to be a living image of the god.

ORACLE OF TROPHONIUS

Whoever consulted this oracle at Lebadea descended by night into a
chasm and returned so shaken that he could not smile for days.

CHAPTER XXXVII

ZOROASTER

Zoroaster taught that Ormuzd the source of good and Ahriman the author
of evil carry on an unending war across the whole of creation.

HINDU MYTHOLOGY

Brahma, Vishnu and Siva are the three chief gods, and the sacred river
Ganges is said to flow from the foot of Vishnu.

BUDDHA

Buddha taught the way of release from desire, and in Tibet the Grand
Lama is honoured as a living continuation of his spirit.

PRESTER JOHN

A letter long believed genuine told of a vast Christian kingdom in the
East ruled by a priest-king of boundless wealth.

CHAPTER XXXVIII

NORTHERN MYTHOLOGY

The Scandinavians told of Odin and the hall of Valhalla, where the
slain feast until the last battle at the end of the world.

CHAPTER XXXIX

THOR'S VISIT TO JOTUNHEIM

Thor journeyed to the giant's country with Loki and Thialfi and lodged
the first night in a hall that proved to be a giant's glove.

CHAPTER XLI

THE DRUIDS

The Druids were the priests of the Celtic nations, and held their
rites within circles of standing stones near a stream or an oak grove.

GLOSSARY

MIDAS. A king of Phrygia whose touch turned all to gold, and who later
received the ears of an ass from Apollo for a poor judgement in music.
"""

_SMALL = dict(chunk_size=25, overlap=0)


def _traditions(volume: str, url: str) -> dict[str, str | None]:
    chunks = split_into_chunks(volume, **_SMALL)
    labels = assign_chunk_traditions(volume, chunks, url)
    return {chunk: label for chunk, label in zip(chunks, labels, strict=True)}


def _tradition_of(labelled: dict[str, str | None], needle: str) -> str | None:
    return next(label for chunk, label in labelled.items() if needle in chunk)


_URL = "https://www.gutenberg.org/cache/epub/4925/pg4925.txt"


def test_source_without_a_table_gets_all_none():
    chunks = ["first chunk of text", "second chunk of text"]
    assert assign_chunk_traditions("body", chunks, "https://www.gutenberg.org/ebooks/1152") == [
        None,
        None,
    ]


def test_greek_chapters_inherit_the_source_tradition():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "Prometheus was one of the Titans") is None
    assert _tradition_of(labelled, "numbers are the principle") is None


def test_mid_chapter_subsection_flips_to_egyptian_and_back():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "worshipped Osiris and Isis") == "Egyptian"
    # the oracles that follow are Greek again
    assert _tradition_of(labelled, "consulted this oracle at Lebadea") is None


def test_eastern_chapter_splits_into_persian_hindu_and_buddhist():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "Ormuzd the source of good") == "Persian"
    assert _tradition_of(labelled, "Brahma, Vishnu and Siva") == "Hindu"
    assert _tradition_of(labelled, "way of release from desire") == "Buddhist"
    assert _tradition_of(labelled, "vast Christian kingdom") is None


def test_northern_chapters_are_norse_through_chapter_xl():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "hall of Valhalla") == "Norse"
    assert _tradition_of(labelled, "journeyed to the giant's country") == "Norse"


def test_druids_chapter_is_celtic():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "priests of the Celtic nations") == "Celtic"


def test_glossary_returns_to_the_source_tradition():
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "king of Phrygia whose touch") is None


def test_contents_list_glossary_does_not_open_a_span_early():
    # GLOSSARY appears in the front-matter contents block above every body
    # heading; if it created a boundary there, the Prometheus chunk would
    # be mislabelled instead of inheriting.
    labelled = _traditions(BULFINCH_VOLUME, _URL)
    assert _tradition_of(labelled, "Prometheus was one of the Titans") is None


def test_chunk_count_is_preserved():
    chunks = split_into_chunks(BULFINCH_VOLUME, **_SMALL)
    assert len(assign_chunk_traditions(BULFINCH_VOLUME, chunks, _URL)) == len(chunks)
