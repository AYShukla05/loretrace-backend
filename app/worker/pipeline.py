import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import split_into_chunks
from app.dedup import content_unchanged, diff_chunks, hash_text
from app.embedding import embed_texts
from app.models.chunk import Chunk
from app.models.source import Source
from app.scraping.fetch import NotModifiedError, fetch_source_text


async def process_source(source: Source, db: AsyncSession) -> bool:
    """Scrape, dedup, chunk, and embed a source's content.

    Returns True if new content was fetched and chunk data changed, False if
    the source was confirmed unchanged (a 304, or a matching content hash) and
    no chunk work was needed.
    """
    async with httpx.AsyncClient() as client:
        try:
            fetch_result = await fetch_source_text(client, source)
        except NotModifiedError:
            return False

    if content_unchanged(source.content_hash, fetch_result.text):
        source.etag = fetch_result.etag
        source.last_modified = fetch_result.last_modified
        return False

    chunk_texts = split_into_chunks(fetch_result.text)
    hash_to_text = {hash_text(text): text for text in chunk_texts}

    existing = await db.execute(
        select(Chunk).where(Chunk.source_id == source.id, Chunk.is_active.is_(True))
    )
    existing_chunks = {chunk.chunk_hash: chunk for chunk in existing.scalars()}

    diff = diff_chunks(existing_chunks.keys(), chunk_texts)

    for stale_hash in diff.stale:
        existing_chunks[stale_hash].is_active = False

    new_hashes = list(diff.new)
    if new_hashes:
        embeddings = await asyncio.to_thread(embed_texts, [hash_to_text[h] for h in new_hashes])
        for chunk_hash, embedding in zip(new_hashes, embeddings, strict=True):
            db.add(
                Chunk(
                    source_id=source.id,
                    chunk_text=hash_to_text[chunk_hash],
                    chunk_hash=chunk_hash,
                    embedding=embedding,
                )
            )

    source.content_hash = hash_text(fetch_result.text)
    source.etag = fetch_result.etag
    source.last_modified = fetch_result.last_modified
    return True
