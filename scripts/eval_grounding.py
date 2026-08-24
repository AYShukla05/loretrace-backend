"""Grounding-attribution eval: does an answer actually come from the retrieved
excerpts, or from what the base model already knew?

The corpus is all public-domain myth texts the base model was trained on, so a
correct answer about Odin proves nothing on its own - the model could produce
it with retrieval switched off entirely. Grounding is only *observable* where
the provided context and the model's pretraining disagree. This script
manufactures that disagreement two ways:

1. Invented-entity probes. A fabricated figure (Runvaldr Tidekeeper) with
   fabricated attributes, handed to the model as context. Nothing about him
   exists in any training data, so a specific, correct answer can only have
   come from the excerpt. A refusal means the excerpt was ignored.

2. Contradiction probes. A well-known Norse fact, deliberately altered in the
   context (Odin with three ravens, Sleipnir with six legs, Gunnar slaying
   Fafnir). Following the altered version proves the answer tracks the
   provided source; giving the real textbook version proves pretraining
   overrode it.

Every chunk here is hand-built and passed straight to generate_answer - no DB,
no retrieval, no Neon, and no change to the live corpus or any app/ code. Same
hand-built-RetrievedChunk pattern this project's earlier one-off Gate 2/3/4
probe scripts used, kept as a committed reusable form.

Each probe runs on both Groq tiers (primary and fallback), and every answer is
scored two ways: a cheap marker check, and a separate LLM grounding judge that
sees only the answer and the exact excerpts it was built from.

Makes real Groq calls - one per probe per tier, plus one judge call each, about
36 in total. It costs nothing in money: Groq's free tier has no card on file and
no metered billing, and going over a limit is a hard 429 until the window
resets, never a charge. It does spend free-tier quota, so the run is paced (a
fixed gap before every call, plus 429 backoff) to stay well under the 30
requests/min and 8,000 tokens/min ceilings and to not starve real /chat traffic
sharing the same key. Confirm before running anyway, same as any other live-API
action in this project.

Two modes:

    python scripts/eval_grounding.py          # offline divergence probes (above)
    python scripts/eval_grounding.py --live   # live end-to-end faithfulness pass

The --live mode is the other half of Gate 6. It runs the real production path
- retrieve_chunks against live Neon, then generate_answer through the real
fallback chain - for Gate 1's 12 ground-truth queries, then judges each answer
against the chunks that were actually retrieved. It answers a different
question: given real (sometimes imperfect) retrieved context, does the answer
stay inside it? Read-only against Neon; no writes, no corpus change, no app/
change. Same free-tier Groq cost profile as the offline mode.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.core.config import settings
from app.llm import LLMError, _call_groq, _format_context, generate_answer
from app.retrieval import RetrievedChunk, retrieve_chunks

if hasattr(sys.stdout, "reconfigure"):
    # Norse names and Groq's typographic punctuation both trip the Windows
    # console's default cp1252 encoding; UTF-8 stdout sidesteps it. Guarded so
    # importing this module under pytest (captured stdout has no reconfigure)
    # doesn't blow up.
    sys.stdout.reconfigure(encoding="utf-8")

TIERS = [("primary", settings.groq_model), ("fallback", settings.groq_fallback_model)]

# Free-tier pacing. The gpt-oss free tier is 30 req/min and 8,000 tokens/min
# (hidden reasoning tokens count too); overage is a 429 until the window resets,
# never a charge. A fixed gap before every call keeps a full ~36-call run near
# 6-7 tokens-per-minute headroom and stops an eval burst from briefly starving
# real /chat traffic on the same key.
CALL_GAP_SECONDS = 8.0
RATE_LIMIT_BACKOFF_SECONDS = 45.0
MAX_CALL_ATTEMPTS = 3

# Phrases that mean the model declined to use the context it was given. For the
# invented-entity probes that is the failure mode: the excerpt was right there.
_IGNORANCE_MARKERS = [
    "no mention",
    "not mentioned",
    "excerpts do not",
    "excerpts don't",
    "does not mention",
    "doesn't mention",
    "could not find",
    "couldn't find",
    "i don't have",
    "i do not have",
    "no information",
    "cannot answer",
    "can't answer",
    "not covered",
    "nothing in the",
    "unable to find",
]

JUDGE_PROMPT = (
    "You are a strict grounding checker. You get source excerpts and an answer "
    "that was supposedly written using only those excerpts. Break the answer "
    "into its distinct factual claims. Mark each SUPPORTED if the excerpts "
    "state or directly imply it, or UNSUPPORTED if they do not (it may still "
    "be true in general - that is not the question). Then set grounded to true "
    "only if every substantive claim is SUPPORTED. Reply with only JSON: "
    '{"claims": [{"text": "...", "verdict": "SUPPORTED" | "UNSUPPORTED"}], '
    '"grounded": true | false}'
)


def _probe_chunk(chunk_id: int, source_id: int, title: str, text: str) -> RetrievedChunk:
    # .invalid is a reserved TLD - a reminder that every source here is synthetic.
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_url=f"https://example.invalid/{source_id}",
        tradition="Norse",
        chunk_text=text,
        distance=0.15,
        title=title,
    )


_TIDEKEEPER_CHUNKS = [
    _probe_chunk(
        90001,
        900,
        "The Kelfjord Tide-Lay",
        "Runvaldr Tidekeeper watches the mouth of the river Kelfjord, where the "
        "tide-runes are cut into nine standing stones. Each morning he reads "
        "them and knows from their wet and dry faces how the sea will run that day.",
    ),
    _probe_chunk(
        90002,
        900,
        "The Kelfjord Tide-Lay",
        "Runvaldr carries three copper horns, named Andvekr, Solmr, and Thveit. "
        "He sounds Andvekr to call the tide in, Solmr to hold it still, and "
        "Thveit to send it out again. An otter called Skeln swims always at his side.",
    ),
]

INVENTED_ENTITY_PROBES: list[dict] = [
    {
        "kind": "invented_entity",
        "name": "invented figure - identity",
        "question": "Who is Runvaldr Tidekeeper, and where does he keep watch?",
        "chunks": _TIDEKEEPER_CHUNKS,
        "grounded_markers": ["Kelfjord"],
        "fail_markers": _IGNORANCE_MARKERS,
        "criteria": (
            "Pass: describes Runvaldr from the excerpt (the river Kelfjord, the "
            "tide-runes). Fail: claims no information on him - the excerpt was "
            "provided and ignored."
        ),
    },
    {
        "kind": "invented_entity",
        "name": "invented figure - fabricated specifics",
        "question": "What are the names of Runvaldr's three copper horns?",
        "chunks": _TIDEKEEPER_CHUNKS,
        "grounded_markers": ["Andvekr", "Solmr", "Thveit"],
        "fail_markers": _IGNORANCE_MARKERS,
        "criteria": (
            "Pass: names Andvekr, Solmr, Thveit. These exist nowhere but the "
            "excerpt, so a correct answer can only be grounded in it. Fail: a "
            "refusal, or invented names not in the excerpt."
        ),
    },
    {
        "kind": "invented_entity",
        "name": "invented figure - companion",
        "question": "What animal travels with Runvaldr Tidekeeper?",
        "chunks": _TIDEKEEPER_CHUNKS,
        "grounded_markers": ["otter", "Skeln"],
        "fail_markers": _IGNORANCE_MARKERS,
        "criteria": "Pass: an otter named Skeln. Fail: a refusal, or a different animal.",
    },
]

_FJALGRIM = "The Fjalgrim Codex"

CONTRADICTION_PROBES: list[dict] = [
    {
        "kind": "contradiction",
        "name": "altered fact - Odin's ravens",
        "question": "How many ravens does Odin have, and what are they called?",
        "chunks": [
            _probe_chunk(
                90101,
                901,
                _FJALGRIM,
                "Odin keeps three ravens, and their names are Huginn, Muninn, and "
                "Verthr. Each dawn he sends all three across the nine lands, and by "
                "nightfall they return to whisper what they have seen into his ear.",
            )
        ],
        "grounded_markers": ["Verthr", "three ravens"],
        "fail_markers": ["two ravens", "pair of ravens", "only two", "two, named", "just two"],
        "criteria": (
            "Pass: follows the excerpt - three ravens, including Verthr. Fail: "
            "'two ravens, Huginn and Muninn', the textbook answer, meaning "
            "pretraining overrode the source."
        ),
    },
    {
        "kind": "contradiction",
        "name": "altered fact - Thor's hammer name",
        "question": "What is the name of Thor's hammer?",
        "chunks": [
            _probe_chunk(
                90102,
                901,
                _FJALGRIM,
                "The hammer of Thor is called Skolvir. It was forged in the deep "
                "places by the dark elves, and when Thor hurls it at his foes it "
                "flies back into his hand of its own accord.",
            )
        ],
        "grounded_markers": ["Skolvir"],
        "fail_markers": ["Mjolnir", "Mjollnir", "Mjölnir"],
        "criteria": (
            "Pass: Skolvir, the name the excerpt gives. Fail: Mjolnir - correct "
            "in general, but not what the provided source says."
        ),
    },
    {
        "kind": "contradiction",
        "name": "altered fact - Sleipnir's legs",
        "question": "How many legs does Sleipnir have?",
        "chunks": [
            _probe_chunk(
                90103,
                901,
                _FJALGRIM,
                "Sleipnir, the grey steed that Odin rides, runs on six legs. No "
                "horse of the giants or of men can match its pace over sea, sky, "
                "or stone.",
            )
        ],
        "grounded_markers": ["six"],
        "fail_markers": ["eight legs", "eight-legged", "eight, "],
        "criteria": "Pass: six, per the excerpt. Fail: eight, the textbook answer.",
    },
    {
        "kind": "contradiction",
        "name": "altered fact - worlds of Yggdrasil",
        "question": "How many worlds does Yggdrasil connect?",
        "chunks": [
            _probe_chunk(
                90104,
                901,
                _FJALGRIM,
                "The great ash Yggdrasil binds together seven worlds. Its roots and "
                "branches hold them each in place, and when the tree shakes, every "
                "one of the seven feels it.",
            )
        ],
        "grounded_markers": ["seven"],
        "fail_markers": ["nine worlds", "nine realms", "nine, "],
        "criteria": "Pass: seven, per the excerpt. Fail: nine, the textbook answer.",
    },
    {
        "kind": "contradiction",
        "name": "altered fact - slayer of Fafnir",
        "question": "Who slew the dragon Fafnir?",
        "chunks": [
            _probe_chunk(
                90105,
                901,
                _FJALGRIM,
                "It was Gunnar who slew the dragon Fafnir. He waited in a pit dug "
                "across the worm's path and drove his blade upward as the great "
                "serpent crawled over him toward the water.",
            )
        ],
        "grounded_markers": ["Gunnar"],
        "fail_markers": ["Sigurd", "Sigurth", "Sigurdr", "Siegfried"],
        "criteria": "Pass: Gunnar, per the excerpt. Fail: Sigurd, the textbook answer.",
    },
    {
        "kind": "contradiction",
        "name": "altered fact - Loki's parents",
        "question": "Who are Loki's parents?",
        "chunks": [
            _probe_chunk(
                90106,
                901,
                _FJALGRIM,
                "Loki is counted among the sons of Odin and Frigg, brother to the "
                "gods of Asgard, though his deeds set him ever apart from that kin.",
            )
        ],
        "grounded_markers": ["Frigg"],
        "fail_markers": ["Farbauti", "Fárbauti", "Laufey", "Nal", "giant"],
        "criteria": (
            "Pass: Odin and Frigg, per the excerpt. Fail: Farbauti and Laufey, "
            "the textbook answer."
        ),
    },
]

ALL_PROBES = INVENTED_ENTITY_PROBES + CONTRADICTION_PROBES


def classify(answer: str, probe: dict) -> str:
    """Cheap first-pass verdict from substring markers. The printed answer and
    the LLM judge are the real signal - this just flags the obvious cases.
    """
    text = answer.lower()
    has_grounded = any(m.lower() in text for m in probe["grounded_markers"])
    has_fail = any(m.lower() in text for m in probe["fail_markers"])

    if probe["kind"] == "contradiction":
        # Naming the real/textbook fact is a leak even if the altered one is
        # also mentioned - the model brought in outside knowledge.
        if has_fail:
            return "fail"
        if has_grounded:
            return "pass"
        return "ambiguous"

    # invented_entity: using any fabricated detail from the excerpt is a pass,
    # even alongside a hedge.
    if has_grounded:
        return "pass"
    if has_fail:
        return "fail"
    return "ambiguous"


def _parse_judge(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}")
        data = json.loads(raw[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return {"grounded": None, "supported": 0, "unsupported": 0, "raw": raw}

    claims = data.get("claims") or []
    supported = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "SUPPORTED")
    unsupported = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "UNSUPPORTED")
    return {
        "grounded": data.get("grounded"),
        "supported": supported,
        "unsupported": unsupported,
        "raw": raw,
    }


async def judge_grounded(
    client: httpx.AsyncClient, model: str, answer: str, chunks: list[RetrievedChunk]
) -> dict:
    """Second opinion: hand the judge only the answer and the exact excerpts it
    was built from, and ask whether every claim traces back to them.
    """
    user_msg = f"EXCERPTS:\n\n{_format_context(chunks)}\n\nANSWER:\n\n{answer}"
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = await _call_groq(client, model, messages)
    return _parse_judge(raw)


async def _paced(label: str, make_call):
    """Wait CALL_GAP_SECONDS, run the call, and retry a 429 after a longer
    backoff. Keeps a full run inside the free-tier per-minute limits without
    ever spending money - overage on that tier is a 429, not a charge.
    """
    for attempt in range(MAX_CALL_ATTEMPTS):
        await asyncio.sleep(CALL_GAP_SECONDS)
        try:
            return await make_call()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and attempt < MAX_CALL_ATTEMPTS - 1:
                print(f"  429 on {label}, backing off {RATE_LIMIT_BACKOFF_SECONDS:.0f}s")
                await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise
    raise AssertionError("unreachable")


async def evaluate(client: httpx.AsyncClient) -> list[dict]:
    rows: list[dict] = []
    for tier_name, model in TIERS:
        print(f"\n{'=' * 70}\nTIER: {tier_name} ({model})\n{'=' * 70}")
        for probe in ALL_PROBES:
            print(f"\n--- {probe['name']} [{probe['kind']}] ---")
            print(f"Q: {probe['question']}")
            print(f"Criteria: {probe['criteria']}")

            row = {"tier": tier_name, "kind": probe["kind"], "name": probe["name"]}
            try:
                answer = await _paced(
                    f"{probe['name']}/{tier_name}",
                    lambda p=probe, m=model: generate_answer(
                        client, p["question"], p["chunks"], model=m
                    ),
                )
                judged = await _paced(
                    f"judge/{probe['name']}/{tier_name}",
                    lambda p=probe, a=answer: judge_grounded(client, TIERS[0][1], a, p["chunks"]),
                )
            except (LLMError, httpx.HTTPStatusError) as exc:
                print(f"Result: LLM ERROR - {exc}")
                row.update(verdict="error", judge=None)
                rows.append(row)
                continue

            verdict = classify(answer, probe)
            print(f"Answer:\n{answer}")
            print(f"Marker verdict: {verdict.upper()}")
            print(
                f"Judge: grounded={judged['grounded']} "
                f"(supported={judged['supported']}, unsupported={judged['unsupported']})"
            )
            row.update(verdict=verdict, judge=judged["grounded"])
            rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for tier_name, _ in TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier_name]
        for kind, label in [
            ("contradiction", "CONTRADICTION (followed the altered source)"),
            ("invented_entity", "INVENTED-ENTITY (used the provided facts)"),
        ]:
            kr = [r for r in tier_rows if r["kind"] == kind]
            n = len(kr)
            passed = sum(1 for r in kr if r["verdict"] == "pass")
            failed = sum(1 for r in kr if r["verdict"] == "fail")
            amb = sum(1 for r in kr if r["verdict"] == "ambiguous")
            err = sum(1 for r in kr if r["verdict"] == "error")
            judged_ok = sum(1 for r in kr if r["judge"] is True)
            print(
                f"\n{tier_name:9s} {label}\n"
                f"          markers : {passed}/{n} pass, {failed} fail, "
                f"{amb} ambiguous, {err} error\n"
                f"          judge   : {judged_ok}/{n} rated fully grounded"
            )
    print(
        "\nHow to read this:\n"
        "  contradiction FAIL = the answer gave the real textbook fact instead of\n"
        "    the one in the excerpt - pretraining is leaking past retrieval.\n"
        "  invented-entity FAIL = the model refused despite being handed the facts\n"
        "    - the provided context is not actually being used.\n"
        "  Anything else on both families is grounding behaving as intended."
    )


async def run_live_faithfulness(client: httpx.AsyncClient) -> None:
    """Gate 6's live half: real retrieval + real answer + judge, for the Gate 1
    ground-truth queries. Retrieval runs first with the DB session open only
    briefly, then the slow Groq work happens with no Neon connection held (same
    discipline as the re-embed and Gate 1 scripts).
    """
    from app.db.session import async_session
    from scripts.eval_gate1_recall import QUERIES as RETRIEVAL_QUERIES

    print(f"\n{'=' * 70}\nLIVE FAITHFULNESS (real retrieve_chunks + generate_answer)\n{'=' * 70}")

    retrieved: list[tuple[str, list[RetrievedChunk]]] = []
    async with async_session() as db:
        for q in RETRIEVAL_QUERIES:
            retrieved.append((q["query"], await retrieve_chunks(db, q["query"])))

    grounded_ok = 0
    refused = 0
    answered = 0
    for question, chunks in retrieved:
        print(f"\n--- {question[:90]} ---")
        if not chunks:
            refused += 1
            print("Result: REFUSED (empty retrieval) - grounded by construction")
            continue

        answered += 1
        try:
            answer = await _paced(
                f"answer/{question[:30]}",
                lambda q=question, c=chunks: generate_answer(client, q, c),
            )
            judged = await _paced(
                f"judge/{question[:30]}",
                lambda c=chunks, a=answer: judge_grounded(client, TIERS[0][1], a, c),
            )
        except (LLMError, httpx.HTTPStatusError) as exc:
            print(f"Result: LLM ERROR - {exc}")
            continue

        if judged["grounded"] is True:
            grounded_ok += 1
        sources = sorted({c.title or c.source_url for c in chunks})
        print(f"Sources retrieved: {sources}")
        print(f"Answer:\n{answer}")
        print(
            f"Judge: grounded={judged['grounded']} "
            f"(supported={judged['supported']}, unsupported={judged['unsupported']})"
        )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"queries           : {len(retrieved)}")
    print(f"refused (empty)    : {refused}")
    print(f"answered           : {answered}")
    print(f"judged grounded    : {grounded_ok}/{answered}")
    print(
        "\nHow to read this: a low grounded count means answers are drifting "
        "past what retrieval actually returned - either into pretraining or "
        "into unsupported synthesis. Empty-retrieval refusals are grounded by "
        "construction and not counted against the rate."
    )


async def main() -> None:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set; cannot run.")
        return
    async with httpx.AsyncClient() as client:
        if "--live" in sys.argv:
            await run_live_faithfulness(client)
        else:
            print_summary(await evaluate(client))


if __name__ == "__main__":
    asyncio.run(main())
