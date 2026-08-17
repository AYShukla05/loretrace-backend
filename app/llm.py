import httpx

from app.core.config import settings
from app.retrieval import RetrievedChunk
from app.self_hosted import call_self_hosted

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
CLOUDFLARE_CHAT_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

# Rules 3 and 4 map directly to LoreTrace_Bias_Mitigation_Plan.md Part 1: a
# model can get an answer "factually right" excerpt by excerpt and still leak
# pretrained bias by synthesizing across sources itself, or by hedging a claim
# a source states plainly. Both are addressed as explicit prompt rules, not
# left to the model's default RLHF-tuned register. The voice itself (warm,
# storyteller, not a lecture) was a deliberate rewrite decided with the user
# 2026-08-15/16 after real usage read as dry and clinical — see CLAUDE.md's
# "citation readability" session notes for the full rationale. Rule 5 depends
# on _format_context naming sources by title rather than a numeric label.
SYSTEM_PROMPT = """You're a storyteller who's spent a lifetime with these old myths and beliefs \
— think of how someone's grandmother might answer when a grandchild asks about the gods and \
legends of their people. Warm, plainspoken, unhurried. Not a professor giving a lecture, and not \
a museum placard reciting dates. Follow these rules exactly:

1. Answer only using the provided excerpts. Never add facts, comparisons, or context from your \
own training knowledge, even if you know more about the topic than the excerpts contain.
2. Treat the beliefs you're retelling as living wisdom, not a specimen under glass. These \
stories carried real meaning for the people who told them — memory, metaphor, moral guidance, a \
way of making sense of the world. Let that meaning come through the way the excerpts themselves \
tell it, rather than flattening everything into "then this happened, then that happened." Don't \
invent meaning the excerpts don't support — just don't strip out the meaning that's plainly \
there in how the source tells the story.
3. No unsupported cross-source synthesis. If answering would require combining claims from two \
or more sources into something neither states on its own — a comparison, a shared-origin theory, \
a causal link — don't make that combination yourself. Say plainly the corpus doesn't have a \
source drawing that connection, unless one already does.
4. No unsourced hedging. Don't add "some scholars believe" framing unless a source itself says \
that. If a source states something directly, tell it directly.
5. Name where a story comes from in plain language, not a citation number — by its actual name \
if you have one, or a short natural description if you don't. Say it once per distinct source, \
worked naturally into the telling, not repeated per sentence. If something about a source's own \
history is worth knowing (who wrote it down, when, any noted bias), share that plainly too, once \
— not as a disclaimer bolted onto every claim.
6. If the excerpts don't answer the question, say so plainly instead of guessing.
7. If sources disagree, tell each version separately, in its own voice, never blended together. \
You can share why they likely differ, if that reason comes from what you actually know about \
each source's own background — who wrote it down, when, from what vantage point — not from \
outside knowledge you're bringing in yourself. But explaining the "why" doesn't mean deciding \
which account is correct. If a source comes from a colonial administrator, missionary, or \
Western academic looking in from outside, say so plainly and let the reader weigh it."""


class LLMError(RuntimeError):
    pass


def _provenance_label(chunk: RetrievedChunk) -> str:
    tags = [
        chunk.author_position.value.replace("_", " ") if chunk.author_position else None,
        chunk.text_role.value.replace("_", " ") if chunk.text_role else None,
        chunk.era.value.replace("_", " ") if chunk.era else None,
    ]
    label = ", ".join(tag for tag in tags if tag)
    if chunk.known_bias_flags:
        flag = f"flagged: {chunk.known_bias_flags}"
        label = f"{label}, {flag}" if label else flag
    return label


def _source_label(chunk: RetrievedChunk) -> str:
    """A name the model can actually cite in prose, per
    LoreTrace_Bias_Mitigation_Plan.md's citation-by-name direction, instead
    of a numbered "Source N" that means nothing to a reader. Most sources
    have a title now that scraping infers one (see app/scraping/fetch.py),
    but older or manually-added sources may not, so a plain description is
    the fallback rather than a hard requirement.
    """
    if chunk.title:
        return chunk.title
    if chunk.tradition:
        return f"an untitled {chunk.tradition} source"
    return "an untitled source"


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Groups excerpts by source_id, not by chunk: a multi-excerpt retrieval
    from a single source (the common case once a corpus has few sources with
    many chunks each) previously numbered every chunk as its own "Source N"
    with a repeated, identical provenance line, which read as several
    different sources to both the model and the reader.
    """
    order: list[int] = []
    blocks: dict[int, list[str]] = {}
    for chunk in chunks:
        if chunk.source_id not in blocks:
            order.append(chunk.source_id)
            label = _source_label(chunk)
            tradition_suffix = f" ({chunk.tradition})" if chunk.tradition and chunk.title else ""
            header = f"[{label}{tradition_suffix}: {chunk.source_url}]"
            provenance = _provenance_label(chunk)
            if provenance:
                header += f"\nProvenance: {provenance}"
            blocks[chunk.source_id] = [header]
        blocks[chunk.source_id].append(chunk.chunk_text)
    return "\n\n".join("\n".join(blocks[source_id]) for source_id in order)


async def _call_groq(client: httpx.AsyncClient, model: str, messages: list[dict]) -> str:
    response = await client.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={"model": model, "messages": messages, "temperature": 0.2},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


async def _call_cloudflare(client: httpx.AsyncClient, messages: list[dict]) -> str:
    url = CLOUDFLARE_CHAT_URL.format(
        account_id=settings.cloudflare_account_id, model=settings.cloudflare_model
    )
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {settings.cloudflare_api_token}"},
        json={"messages": messages},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["result"]["response"]


def _fallback_tiers(client: httpx.AsyncClient, messages: list[dict]) -> list:
    tiers = [
        lambda: _call_groq(client, settings.groq_model, messages),
        lambda: _call_groq(client, settings.groq_fallback_model, messages),
    ]
    if settings.cloudflare_account_id and settings.cloudflare_api_token:
        tiers.append(lambda: _call_cloudflare(client, messages))
    if settings.self_hosted_url:
        tiers.append(lambda: call_self_hosted(client, messages))
    return tiers


async def generate_stock_answer(client: httpx.AsyncClient, query: str) -> str:
    """The comparison-mode baseline: no system prompt, no retrieved context,
    just the question straight to the primary Groq model, so the answer
    reflects pretraining alone per LoreTrace_Bias_Mitigation_Plan.md Part 5.
    Deliberately skips the fallback chain — this is a demo of a specific
    named model, not a resilience-critical path, so a 429 should surface as
    an honest failure rather than silently substituting a smaller model.
    """
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not configured")

    try:
        return await _call_groq(client, settings.groq_model, [{"role": "user", "content": query}])
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"stock model request failed: {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise LLMError(f"stock model request failed: {exc}") from exc


async def generate_answer(
    client: httpx.AsyncClient,
    query: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
) -> str:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not configured")

    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{context}\n\nQuestion: {query}"},
    ]

    if model is not None:
        return await _call_groq(client, model, messages)

    tiers = _fallback_tiers(client, messages)
    for i, call in enumerate(tiers):
        try:
            return await call()
        except httpx.HTTPStatusError as exc:
            # 429 is the original rate-limit case. 404 is a real incident, not
            # a hypothetical: Groq retired llama-3.3-70b/llama-3.1-8b entirely
            # on 2026-08-17 with no warning to this app, and the primary tier
            # returned 404 instead of 429, propagating as an unhandled error
            # with a working fallback chain sitting right there unused. Other
            # 4xx/5xx codes still raise immediately — those indicate a
            # malformed request or a provider outage, not "this specific
            # model is gone," and shouldn't be silently masked by falling
            # through to a different tier.
            if exc.response.status_code not in (429, 404) or i == len(tiers) - 1:
                raise
    raise AssertionError("unreachable")
