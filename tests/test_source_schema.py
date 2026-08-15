import pytest

from app.schemas.source import SourceCreate, SourceUpdate, normalize_tradition


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("norse", "Norse"),
        ("NORSE", "Norse"),
        ("  Norse  ", "Norse"),
        ("greek   mythology", "Greek Mythology"),
        (None, None),
        ("   ", None),
        ("", None),
    ],
)
def test_normalize_tradition(raw, expected):
    assert normalize_tradition(raw) == expected


def test_source_create_normalizes_tradition():
    source = SourceCreate(url="https://example.com/a", tradition="  norse  ")
    assert source.tradition == "Norse"


def test_source_update_normalizes_tradition():
    source = SourceUpdate(tradition="NORSE")
    assert source.tradition == "Norse"
