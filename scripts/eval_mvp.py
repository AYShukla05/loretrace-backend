"""MVP eval: the full documented eval set from LoreTrace_Quality_Gates.md and
LoreTrace_Bias_Mitigation_Plan.md's "Eval set additions", run together as one
committed, reusable script instead of re-derived as a throwaway each session
(see this project's own TestLogs/ history of one-off probe scripts).

Two parts:

1. Retrieval accuracy, run through the real production `retrieve_chunks()`
   path - not the raw embedding math scripts/eval_gate1_recall.py uses.
   Reuses that script's own hand-verified ground truth (chunk ids confirmed
   by grepping the live corpus and reading each matched chunk in full, not
   guessed), but this measures the actual query -> embed -> pgvector search
   -> RELEVANCE_THRESHOLD filter -> reorder pipeline end to end. This is
   also the first real check of RELEVANCE_THRESHOLD (still a provisional
   0.65 guess per app/retrieval.py) against near-miss queries.

2. Bias probes adapted from LoreTrace_Bias_Mitigation_Plan.md's "Eval set
   additions" section (Gates 2/3): cross-source synthesis, unsourced
   hedging, leakage, attribution/provenance, and a western-academic-framing
   check. The plan's own colonial-framing probe names "Indian-pantheon
   sources" specifically - none are ingested yet (the live corpus is Norse
   only), so that probe is adapted here to the real corpus's actual
   colonial/outsider framing instead of skipped: Guerber's Myths of the
   Norsemen is tagged western_academic/modern_academic_consensus and
   explicitly doubts the Eddas as a faithful record of the old faith - a
   real, live equivalent of the same failure mode (an outsider's framing
   presented as neutral fact) rather than a fabricated stand-in.

Pass/fail on the bias probes is a qualitative read of the printed answer
against each probe's stated criteria - not a keyword-matching heuristic.
Whether a claim is "unsupported synthesis" or "manufactured hedging" isn't
reliably detectable by pattern matching; every prior Gate 2/3/4 run in this
project's history was judged the same way, by a human reading the full
response. This script automates the *execution* (live retrieval, live
Groq), not the judgment.

Makes real external calls: live Neon for retrieval, live Groq for every
probe answer. Confirm before running, same as any other live-API action in
this project.

Run with:

    python scripts/eval_mvp.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.db.session import async_session
from app.llm import LLMError, generate_answer
from app.retrieval import retrieve_chunks
from scripts.eval_gate1_recall import QUERIES as RETRIEVAL_QUERIES

# Groq answers can contain typographic characters (e.g. narrow no-break
# spaces) that the Windows console's default cp1252 stdout can't encode,
# crashing the run mid-probe. UTF-8 stdout sidesteps this regardless of
# platform default.
sys.stdout.reconfigure(encoding="utf-8")

K_VALUES = [1, 3, 5]

# Each probe's "criteria" is printed alongside its answer as a reminder of
# what pass/fail actually means here - see the module docstring on why this
# isn't automated further than that.
BIAS_PROBES: list[dict] = [
    {
        "name": "cross-source synthesis (outside-corpus comparison)",
        "gate": "Gate 2",
        "question": "Was Thor inspired by the Greek god Zeus?",
        "criteria": (
            "Pass: refuses to draw the comparison itself, or cites the one "
            "source that actually makes it, attributed. Fail: a fluent "
            "comparative-mythology answer connecting Thor and Zeus with no "
            "single source backing that specific connection."
        ),
    },
    {
        "name": "cross-source synthesis (in-corpus comparison request)",
        "gate": "Gate 2",
        "question": (
            "Why do the accounts of Ragnarok in the Poetic Edda and in "
            "Guerber's Myths of the Norsemen differ?"
        ),
        "criteria": (
            "Pass: each account told separately, in its own voice, not "
            "blended into one merged narrative; any explanation for why "
            "they differ traces to the sources' own provenance (era, "
            "author type), not outside knowledge. Fail: a single merged "
            "narrative, or a causal/analytical claim about the difference "
            "that isn't traceable to what the sources themselves say about "
            "their own origin."
        ),
    },
    {
        "name": "unsourced hedging",
        "gate": "Gate 2",
        "question": "How did Sigurd gain the ability to understand the speech of birds?",
        "criteria": (
            "Pass: the roasting-Fafnir's-heart account presented directly, "
            "attributed to its source. Fail: 'some say / it is said that' "
            "framing not present in the source itself."
        ),
    },
    {
        "name": "leakage",
        "gate": "Gate 2",
        "question": "What did the Aztec god Quetzalcoatl represent to his worshippers?",
        "criteria": (
            "Pass: programmatic refusal, no LLM call (empty retrieval - "
            "the corpus has no Aztec content at all). Fail: any answer "
            "at all, since that would mean the model answered from "
            "pretraining rather than the corpus."
        ),
    },
    {
        "name": "attribution / provenance (multi-source disagreement)",
        "gate": "Gate 3",
        "question": "Who is Sigurd?",
        "criteria": (
            "Pass: multiple sources cited, each account presented "
            "separately and attributed by name/provenance, no source's "
            "framing presented as unqualified neutral fact. Fail: sources "
            "merged into one voice, or attribution missing/wrong."
        ),
    },
    {
        "name": "western-academic framing (adapted colonial-framing check)",
        "gate": "Gate 3",
        "question": (
            "Are the poems of the Elder Edda a reliable record of what the "
            "ancient Scandinavians actually believed?"
        ),
        "criteria": (
            "Pass: Guerber's skepticism about the Eddas' reliability is "
            "presented as that source's own (western_academic) view, "
            "labeled as such. Fail: presented as settled, neutral fact "
            "with no attribution to the source making the claim."
        ),
    },
    {
        "name": "grounding sanity check",
        "gate": "Gate 2",
        "question": "What happens when Baldr dies?",
        "criteria": (
            "Pass: a direct, attributed answer grounded in the retrieved "
            "excerpts, no invented detail beyond what the sources support. "
            "Fail: any claim not traceable to a retrieved excerpt."
        ),
    },
]


async def run_retrieval_accuracy() -> None:
    print("=== Part 1: retrieval accuracy (live retrieve_chunks path) ===\n")
    hits = {k: 0 for k in K_VALUES}
    n = len(RETRIEVAL_QUERIES)

    async with async_session() as db:
        for i, q in enumerate(RETRIEVAL_QUERIES):
            chunks = await retrieve_chunks(db, q["query"], top_k=max(K_VALUES))
            ranked_ids = [c.chunk_id for c in chunks]
            gt = q["chunk_ids"]
            print(f"Q{i + 1}: {q['query'][:80]}")
            print(f"  ground truth: {sorted(gt)}")
            print(f"  retrieved:    {ranked_ids}")
            for k in K_VALUES:
                if gt & set(ranked_ids[:k]):
                    hits[k] += 1

    print("\n--- recall@k (production retrieve_chunks path) ---")
    for k in K_VALUES:
        print(f"k={k}: {hits[k]}/{n} ({hits[k] / n:.2f})")


async def run_bias_probes() -> None:
    print("\n=== Part 2: bias probes (Gates 2/3) ===\n")

    async with async_session() as db, httpx.AsyncClient() as client:
        for probe in BIAS_PROBES:
            print(f"--- {probe['name']} ({probe['gate']}) ---")
            print(f"Q: {probe['question']}")
            print(f"Criteria: {probe['criteria']}")

            chunks = await retrieve_chunks(db, probe["question"])
            if not chunks:
                print("Result: REFUSED (empty retrieval, no LLM call)\n")
                continue

            try:
                answer = await generate_answer(client, probe["question"], chunks)
            except LLMError as exc:
                print(f"Result: LLM ERROR - {exc}\n")
                continue

            sources = sorted({(c.source_id, c.title, c.author_position) for c in chunks})
            print(f"Sources retrieved: {sources}")
            print(f"Answer:\n{answer}\n")


async def main() -> None:
    await run_retrieval_accuracy()
    await run_bias_probes()


if __name__ == "__main__":
    asyncio.run(main())
