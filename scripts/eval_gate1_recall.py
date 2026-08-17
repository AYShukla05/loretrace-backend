"""Gate 1 recall@k: self-hosted MiniLM embeddings vs. the Voyage AI baseline,
run against the real ingested corpus. See LoreTrace_Quality_Gates.md Gate 1.

Ground truth chunk ids were built by grepping the live corpus for distinctive
phrases and reading each matched chunk in full, not guessed from memory or
public-benchmark assumptions. A query counts as a hit at k if any of its
ground-truth chunk ids appear in that embedding's top-k ranked chunks.

Re-run whenever the chunking strategy changes (chunk size/overlap affects
retrieval too), not just once and forgotten.

Run with:

    python scripts/eval_gate1_recall.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import select

from app.db.session import async_session
from app.embedding import embed_texts
from app.models.chunk import Chunk

load_dotenv()

VOYAGE_MODEL = "voyage-3-lite"
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_BATCH_SIZE = 50
K_VALUES = [1, 3, 5]

QUERIES: list[dict] = [
    {
        "query": (
            "How did Loki give birth to Odin's eight-legged horse Sleipnir by "
            "turning into a mare and luring away the stallion Svadilfari?"
        ),
        "chunk_ids": {1783},
    },
    {
        "query": (
            "How did the gods create the wise being Kvasir out of their own "
            "saliva after making peace between the Aesir and the Vanir?"
        ),
        "chunk_ids": {1676},
    },
    {
        "query": (
            "Who stole Idun away to a giant's home, and what apples did she "
            "guard that kept the gods young?"
        ),
        "chunk_ids": {1120, 907},
    },
    {
        "query": (
            "What happened when Sigurd burned his finger roasting Fafnir's "
            "heart and tasted the dragon's blood, gaining the power to "
            "understand birds?"
        ),
        "chunk_ids": {609, 1968, 1848},
    },
    {
        "query": (
            "How did the war between the Aesir and the Vanir end, with Njord "
            "and Frey sent as hostages to make peace?"
        ),
        "chunk_ids": {927},
    },
    {
        "query": (
            "How did the gods bind Loki using the entrails of his own son after Baldr's death?"
        ),
        "chunk_ids": {608},
    },
    {
        "query": "What is the mead of poetry, and how did Odin win it from the giant Suttung?",
        "chunk_ids": {51, 784, 846},
    },
    {
        "query": (
            "How does the rainbow bridge Bifrost relate to Ragnarok and the "
            "coming of the people of Muspell?"
        ),
        "chunk_ids": {399},
    },
    {
        "query": (
            "What happened when Sigurd gave Gudrun some of Fafnir's heart to eat, "
            "changing her nature?"
        ),
        "chunk_ids": {2031},
    },
    {
        "query": (
            "What happened when the Volsungs and King Granmar's forces "
            "gathered thousands of warriors for battle?"
        ),
        "chunk_ids": {1438},
    },
    {
        "query": (
            "What does the saga say about Brynhild's awakening and wisdom "
            "after the slaying of the dragon, following the Lay of Fafnir?"
        ),
        "chunk_ids": {1390},
    },
    {
        "query": "Who is Sigrdrifa, the valkyrie connected to Sigurd's story?",
        "chunk_ids": {28, 69, 216, 217, 423, 439, 487, 806, 1390, 1514, 1570, 1620},
    },
]


async def voyage_embed(
    client: httpx.AsyncClient, texts: list[str], input_type: str
) -> list[list[float]]:
    api_key = os.environ["VOYAGE_API_KEY"]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), VOYAGE_BATCH_SIZE):
        batch = texts[i : i + VOYAGE_BATCH_SIZE]
        for attempt in range(3):
            resp = await client.post(
                VOYAGE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": batch, "model": VOYAGE_MODEL, "input_type": input_type},
                timeout=60.0,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  429 from Voyage, backing off {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()
        data = resp.json()
        ordered = sorted(data["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in ordered)
        print(f"  embedded {min(i + VOYAGE_BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors


def cosine_sim_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    q = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    c = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    return q @ c.T


async def main() -> None:
    async with async_session() as db:
        rows = (
            await db.execute(
                select(Chunk.id, Chunk.chunk_text, Chunk.embedding).where(Chunk.is_active.is_(True))
            )
        ).all()
    chunk_ids = [r[0] for r in rows]
    chunk_texts = [r[1] for r in rows]
    self_hosted_matrix = np.array([r[2] for r in rows])

    print(f"Corpus: {len(chunk_ids)} active chunks")

    query_texts = [q["query"] for q in QUERIES]
    self_hosted_query_vecs = np.array(embed_texts(query_texts))
    # embed_texts L2-normalizes, and the stored chunk embeddings were produced
    # the same way at ingest time, so a plain dot product is cosine similarity.
    self_hosted_sims = self_hosted_query_vecs @ self_hosted_matrix.T

    async with httpx.AsyncClient() as client:
        print(f"Embedding {len(chunk_texts)} chunks with Voyage ({VOYAGE_MODEL})...")
        voyage_corpus_vecs = np.array(await voyage_embed(client, chunk_texts, "document"))
        print(f"Embedding {len(query_texts)} queries with Voyage...")
        voyage_query_vecs = np.array(await voyage_embed(client, query_texts, "query"))
    voyage_sims = cosine_sim_matrix(voyage_query_vecs, voyage_corpus_vecs)

    hits = {k: {"self_hosted": 0, "voyage": 0} for k in K_VALUES}
    max_k = max(K_VALUES)
    for qi, q in enumerate(QUERIES):
        gt = q["chunk_ids"]
        sh_ranked = [chunk_ids[i] for i in np.argsort(-self_hosted_sims[qi])]
        vo_ranked = [chunk_ids[i] for i in np.argsort(-voyage_sims[qi])]
        print(f"\nQ{qi + 1}: {q['query'][:80]}")
        print(f"  ground truth: {sorted(gt)}")
        print(f"  self-hosted top-{max_k}: {sh_ranked[:max_k]}")
        print(f"  voyage      top-{max_k}: {vo_ranked[:max_k]}")
        for k in K_VALUES:
            hits[k]["self_hosted"] += bool(gt & set(sh_ranked[:k]))
            hits[k]["voyage"] += bool(gt & set(vo_ranked[:k]))

    n = len(QUERIES)
    print("\n=== Gate 1 recall@k ===")
    for k in K_VALUES:
        sh = hits[k]["self_hosted"]
        vo = hits[k]["voyage"]
        print(f"k={k}: self-hosted={sh}/{n} ({sh / n:.2f})  voyage={vo}/{n} ({vo / n:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
