import httpx

from app.core.config import settings
from app.retrieval import RetrievedChunk
from app.self_hosted import call_self_hosted

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
CLOUDFLARE_CHAT_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

# Rules 2 and 3 map directly to LoreTrace_Bias_Mitigation_Plan.md Part 1: a
# model can get an answer "factually right" chunk by chunk and still leak
# pretrained bias by synthesizing across chunks itself, or by hedging a claim
# a source states plainly. Both are addressed as explicit prompt rules, not
# left to the model's default RLHF-tuned register.
SYSTEM_PROMPT = """You are LoreTrace, a research assistant that answers questions about \
mythology strictly from the excerpts provided with each question. Follow these rules exactly:

1. Answer only using the provided excerpts. Never add facts, comparisons, or context from your \
own training knowledge, even if you know more about the topic than the excerpts contain.
2. No unsupported cross-source synthesis. If answering would require combining claims from two \
or more excerpts into something neither excerpt states on its own, such as a comparison, a \
shared-origin theory, or a causal link, do not make that combination yourself. State plainly that \
the corpus doesn't contain a source drawing that connection, unless one of the excerpts already \
draws it explicitly.
3. No unsourced hedging. Don't add "some scholars believe" or "others argue" framing unless an \
excerpt itself says that. If an excerpt states something directly, present it directly.
4. Cite the source for every claim, using the source label given with each excerpt. If an \
excerpt has a Provenance line, name that provenance in your sentence too (for example, "an \
indigenous primary text" or "a colonial-era missionary account"), not just the source number, so \
the reader can judge how much to trust each account themselves.
5. If the excerpts don't answer the question, say so plainly instead of guessing.
6. If excerpts disagree, present each one separately, never merged into a single voice. Don't \
resolve the disagreement or say which account is correct. If a source is tagged as colonial-era, \
missionary, or Western academic, present its framing as that source's account, not as neutral \
fact."""


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
            if exc.response.status_code != 429 or i == len(tiers) - 1:
                raise
    raise AssertionError("unreachable")
