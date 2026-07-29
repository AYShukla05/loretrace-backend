import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def content_unchanged(previous_hash: str | None, new_text: str) -> bool:
    return previous_hash is not None and previous_hash == hash_text(new_text)


@dataclass(frozen=True)
class ChunkDiff:
    new: frozenset[str]
    stale: frozenset[str]
    unchanged: frozenset[str]


def diff_chunks(existing_hashes: Iterable[str], new_chunk_texts: Iterable[str]) -> ChunkDiff:
    existing = set(existing_hashes)
    incoming = {hash_text(text) for text in new_chunk_texts}
    return ChunkDiff(
        new=frozenset(incoming - existing),
        stale=frozenset(existing - incoming),
        unchanged=frozenset(existing & incoming),
    )
