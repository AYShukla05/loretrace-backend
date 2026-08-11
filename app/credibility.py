import json
import re
import unicodedata
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.credibility_entity import CredibilityEntity
from app.models.enums import AuthorEpistemicBasis, AuthorOrigin, CredibilityEntityType

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Fields an admin might plausibly document for an author or institution. Not
# every field applies to every entity_type (e.g. practice_lineage rarely
# applies to an institution) - the model is told to use null rather than
# guess when a field isn't covered by the pasted text.
#
# birth_region and practice_regions are deliberately separate facts, not one
# combined residence_regions list: LoreTrace_Credibility_Suggestion_Design.md
# section 7 derives author_origin from birth residence and
# author_epistemic_basis from adulthood/practice residence specifically (the
# Max Muller vs. David Frawley contrast in section 4.1 - both foreign-born,
# only one a lived practitioner - only works if those two are distinguishable
# facts, not the same list read two ways).
FACT_KEYS = (
    "birth_year",
    "death_year",
    "birth_region",
    "practice_regions",
    "occupation",
    "institutional_affiliations",
    "practice_lineage",
    "documented_critique_refs",
)

# Only requested when a tradition is supplied. Both answer a question that's
# meaningless without a specific tradition to compare against: e.g. Max
# Muller lived and worked in England his whole career (a real practice_regions
# fact) but never visited India, while David Frawley also lived abroad but is
# a lived Vedic practitioner - practice_regions alone can't distinguish them,
# only whether the text documents actual engagement with the named tradition
# specifically. See LoreTrace_Credibility_Suggestion_Design.md section 4.1.
TRADITION_FACT_KEYS = ("born_within_tradition_context", "tradition_engagement")


def _fact_keys(tradition: str | None) -> tuple[str, ...]:
    return (*FACT_KEYS, *TRADITION_FACT_KEYS) if tradition else FACT_KEYS


def _build_extraction_system_prompt(tradition: str | None) -> str:
    fact_keys = _fact_keys(tradition)
    tradition_clause = ""
    if tradition:
        tradition_clause = f"""
5. "born_within_tradition_context" is true if the text states the person was born or raised in \
the region/community the "{tradition}" tradition originates from, false if the text states a \
birth/upbringing region that is clearly not part of that tradition's own origin - you may use \
general knowledge of which region a named tradition is associated with to make this comparison, \
since that's knowledge about the tradition, not an invented fact about the person. Use null only \
if the text doesn't state where the person was born or raised at all.
6. "tradition_engagement" is true only if the text states the person directly lived within, \
practiced, or otherwise engaged with the "{tradition}" tradition or its community later in life \
(not just general residence or employment elsewhere), false only if the text explicitly states \
they did not (e.g. "never visited"), and null if the text doesn't address it."""

    # Path 1 of LoreTrace_Credibility_Suggestion_Design.md section 5: the model
    # is used in a strictly extractive role. It must never add anything from
    # its own training knowledge about the entity, only what the pasted text
    # states - this is the same "no facts beyond what's given" discipline
    # app/llm.py's SYSTEM_PROMPT enforces on chat answers, applied one layer
    # earlier. Rules 5/6 above extend that discipline to tradition-relative
    # facts: the model still only reads what the pasted text says, it isn't
    # asked to recall anything about the named entity from training data.
    return f"""You extract structured facts about a historical or contemporary \
person or institution from a piece of text an admin has pasted. Follow these rules exactly:

1. Use only facts stated in the pasted text. Never add anything from your own training knowledge \
about the named entity, even if you recognize them and know more.
2. If the text doesn't state a fact, its value is null. Do not guess or infer beyond what the text \
says.
3. Never assert a verdict on a contested claim (for example, whether a text or tradition is \
historically accurate). Only extract biographical and institutional facts.
4. Respond with a single JSON object with exactly these keys: {", ".join(fact_keys)}. \
"birth_year" and "death_year" are integers or null. "birth_region" is the region/country the \
person was born or raised in, as a string or null. "practice_regions" is where the person lived, \
worked, or practiced later in life, as an array of strings or null - this can differ from \
birth_region and both should reflect only what the text actually says. \
"institutional_affiliations" and "documented_critique_refs" are arrays of strings, or null. \
"occupation" and "practice_lineage" are strings or null.{tradition_clause}"""


class ExtractionError(RuntimeError):
    pass


async def extract_facts_from_text(
    client: httpx.AsyncClient,
    entity_type: CredibilityEntityType,
    display_name: str,
    pasted_text: str,
    tradition: str | None = None,
) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise ExtractionError("GROQ_API_KEY is not configured")

    user_content = f"Entity type: {entity_type.value}\nEntity name: {display_name}\n"
    if tradition:
        user_content += f"Tradition being classified: {tradition}\n"
    user_content += f"\nText:\n{pasted_text}"

    response = await client.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            # A structured-extraction task doesn't need the 70B tier's
            # quota - reserve that for actual chat answers.
            "model": settings.groq_fallback_model,
            "messages": [
                {"role": "system", "content": _build_extraction_system_prompt(tradition)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"model did not return valid JSON: {raw!r}") from exc

    # Only ever return the known schema, even if the model adds extra keys.
    return {key: parsed.get(key) for key in _fact_keys(tradition)}


def _build_search_query(entity_type: CredibilityEntityType, display_name: str) -> str:
    if entity_type == CredibilityEntityType.INSTITUTION:
        return f"{display_name} history publisher institution"
    return f"{display_name} biography"


async def search_tavily(
    client: httpx.AsyncClient, query: str, max_results: int = 5
) -> list[dict[str, str]]:
    """Path 2 of LoreTrace_Credibility_Suggestion_Design.md section 5: a
    general web search, not any single encyclopedia, so under-documented
    entities aren't structurally disadvantaged the way an optional
    Wikidata-only lookup would have been (see the doc's decision log).
    """
    if not settings.tavily_api_key:
        raise ExtractionError("TAVILY_API_KEY is not configured")

    response = await client.post(
        TAVILY_SEARCH_URL,
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ExtractionError(f"no search results found for {query!r}")
    return [
        {"url": r["url"], "title": r.get("title", ""), "content": r.get("content", "")}
        for r in results
    ]


def _build_search_extraction_system_prompt(tradition: str | None) -> str:
    fact_keys = _fact_keys(tradition)
    tradition_clause = ""
    if tradition:
        tradition_clause = f"""
5. "born_within_tradition_context" is true if an excerpt states the person was born or raised in \
the region/community the "{tradition}" tradition originates from, false if an excerpt states a \
birth/upbringing region that is clearly not part of that tradition's own origin - you may use \
general knowledge of which region a named tradition is associated with to make this comparison, \
since that's knowledge about the tradition, not an invented fact about the person. Use null only \
if no excerpt states where the person was born or raised at all.
6. "tradition_engagement" is true only if an excerpt states the person directly lived within, \
practiced, or otherwise engaged with the "{tradition}" tradition or its community later in life \
(not just general residence or employment elsewhere), false only if an excerpt explicitly states \
they did not (e.g. "never visited"), and null if no excerpt addresses it."""

    # Same extractive-only discipline as _build_extraction_system_prompt, applied
    # to numbered search excerpts instead of one pasted block. The extra "source"
    # field per fact is what lets fact_provenance carry a real, per-fact URL
    # (design doc section 5) rather than one blanket tag for the whole lookup.
    return f"""You extract structured facts about a historical or contemporary \
person or institution from a set of numbered web search result excerpts. Follow these rules exactly:

1. Use only facts stated in the excerpts. Never add anything from your own training knowledge \
about the named entity, even if you recognize them and know more.
2. If no excerpt states a fact, its value is null. Do not guess or infer beyond what the excerpts \
say.
3. Never assert a verdict on a contested claim (for example, whether a text or tradition is \
historically accurate). Only extract biographical and institutional facts.
4. Respond with a single JSON object with exactly one key, "facts", whose value is an object with \
exactly these keys: {", ".join(fact_keys)}. Each fact is either null, or an object with "value" \
(the fact itself, using the same types described below) and "source" (the integer number of the \
result excerpt that states it). "birth_year" and "death_year" values are integers. "birth_region" \
is a string. "practice_regions" is an array of strings. "institutional_affiliations" and \
"documented_critique_refs" are arrays of strings. "occupation" and "practice_lineage" are \
strings.{tradition_clause}"""


async def extract_facts_from_search_results(
    client: httpx.AsyncClient,
    entity_type: CredibilityEntityType,
    display_name: str,
    results: list[dict[str, str]],
    tradition: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Path 2's extraction step: one LLM call over every search excerpt at once
    (cheaper than one call per result), with each fact tagged by which excerpt
    it came from so fact_provenance can carry a real URL, not just a tier
    label. A fact whose reported source index doesn't map to an actual result
    is dropped entirely rather than kept unattributed - design doc section 2
    requires every suggestion be traceable to a visible source, so an
    unverifiable fact is worth less than no fact.
    """
    if not settings.groq_api_key:
        raise ExtractionError("GROQ_API_KEY is not configured")

    excerpts = "\n\n".join(
        f"[Result {i}] URL: {r['url']}\n{r['content']}" for i, r in enumerate(results, start=1)
    )
    user_content = f"Entity type: {entity_type.value}\nEntity name: {display_name}\n"
    if tradition:
        user_content += f"Tradition being classified: {tradition}\n"
    user_content += f"\nSearch result excerpts:\n{excerpts}"

    response = await client.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_fallback_model,
            "messages": [
                {"role": "system", "content": _build_search_extraction_system_prompt(tradition)},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"model did not return valid JSON: {raw!r}") from exc

    raw_facts = parsed.get("facts") or {}
    url_by_index = {i: r["url"] for i, r in enumerate(results, start=1)}

    facts: dict[str, Any] = {}
    fact_provenance: dict[str, str] = {}
    for key in _fact_keys(tradition):
        entry = raw_facts.get(key)
        source_index = entry.get("source") if isinstance(entry, dict) else None
        url = url_by_index.get(source_index) if isinstance(source_index, int) else None
        if not isinstance(entry, dict) or entry.get("value") is None or url is None:
            facts[key] = None
            continue
        facts[key] = entry["value"]
        fact_provenance[key] = url

    return facts, fact_provenance


def _suggest_author_epistemic_basis(facts: dict[str, Any]) -> dict[str, Any]:
    """First-pass rule table for section 7's author_epistemic_basis row.
    tradition_engagement (present only when a tradition was supplied at
    extraction time) takes priority over practice_lineage alone, since it's
    what actually distinguishes lived practice from textual-only study - see
    the Muller/Frawley contrast noted above TRADITION_FACT_KEYS. Not
    exhaustive, extend as real entities demand it, same as this project's
    other enums were.
    """
    practice_lineage = facts.get("practice_lineage")
    tradition_engagement = facts.get("tradition_engagement")

    if tradition_engagement is False:
        return {
            "value": AuthorEpistemicBasis.TEXTUAL_STUDY_ONLY.value,
            "reasoning": (
                "the pasted text explicitly documents no direct engagement with this tradition"
            ),
            "based_on_facts": ["tradition_engagement"],
        }
    if practice_lineage:
        based_on = ["practice_lineage"]
        if tradition_engagement:
            based_on.append("tradition_engagement")
        return {
            "value": AuthorEpistemicBasis.LIVED_PRACTICE.value,
            "reasoning": "practice_lineage documents a direct lineage or tradition of practice",
            "based_on_facts": based_on,
        }
    if tradition_engagement:
        return {
            "value": AuthorEpistemicBasis.MIXED.value,
            "reasoning": (
                "the pasted text documents direct engagement with this tradition, "
                "but no explicit practice_lineage"
            ),
            "based_on_facts": ["tradition_engagement"],
        }
    return {
        "value": AuthorEpistemicBasis.UNKNOWN.value,
        "reasoning": "insufficient facts to determine epistemic basis",
        "based_on_facts": [],
    }


def _suggest_author_origin(facts: dict[str, Any]) -> dict[str, Any]:
    """section 7's author_origin row. Only called when a tradition was
    supplied (see generate_suggested_values) - "indigenous" is meaningless
    without a tradition to be indigenous relative to, so this deliberately
    doesn't try to infer it from birth_region alone.
    """
    born_within = facts.get("born_within_tradition_context")
    if born_within is True:
        return {
            "value": AuthorOrigin.INDIGENOUS_BORN.value,
            "reasoning": (
                "the pasted text states the person was born/raised within this tradition's "
                "own community or region"
            ),
            "based_on_facts": ["born_within_tradition_context"],
        }
    if born_within is False:
        return {
            "value": AuthorOrigin.FOREIGN_BORN.value,
            "reasoning": (
                "the pasted text states the person was born/raised outside this tradition's "
                "own community or region"
            ),
            "based_on_facts": ["born_within_tradition_context"],
        }
    return {
        "value": AuthorOrigin.UNKNOWN.value,
        "reasoning": "insufficient facts to determine origin relative to this tradition",
        "based_on_facts": [],
    }


def generate_suggested_values(
    entity_type: CredibilityEntityType,
    facts: dict[str, Any],
    tradition: str | None,
) -> dict[str, Any]:
    """Rule-table suggestion generation, section 7. Only author_origin and
    author_epistemic_basis are implemented so far:
    - era needs Era enum year boundaries that don't exist anywhere in this
      codebase yet, plus a decision on whether to bucket by author lifespan
      (a proxy) or an actual publication year this mechanism doesn't
      capture - not attempted here rather than guessed.
    - author_position and historiographical_method need real rule-table
      content written fact-by-fact, which section 12 of the design doc
      itself already flags as not yet decided - not attempted here either.
    Neither field applies to an institution the way they apply to a person,
    so entity_type other than AUTHOR gets no suggestions yet.
    """
    if entity_type != CredibilityEntityType.AUTHOR:
        return {}

    suggested: dict[str, Any] = {
        "author_epistemic_basis": _suggest_author_epistemic_basis(facts),
    }
    if tradition:
        suggested["author_origin"] = _suggest_author_origin(facts)
    return suggested


def normalize_entity_key(display_name: str) -> str:
    """Aggressive normalization for cache dedup, per
    LoreTrace_Credibility_Suggestion_Design.md section 6: lowercase, strip
    diacritics and punctuation, collapse whitespace. Fuzzy near-match
    handling ("did you mean X?") is deferred, not built here - this only
    prevents exact-after-normalization duplicates.
    """
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", display_name) if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


async def get_or_create_credibility_entity(
    db: AsyncSession,
    client: httpx.AsyncClient,
    entity_type: CredibilityEntityType,
    display_name: str,
    pasted_text: str | None = None,
    tradition: str | None = None,
) -> tuple[CredibilityEntity, bool]:
    """Cache-first per section 6: an existing row for this normalized name is
    returned as-is, with no extraction call. Returns (entity, cache_hit).

    Known limitation, not addressed by the original design doc: a cache hit
    returns whatever suggested_values were generated the first time this
    entity was looked up, which may have used a different tradition (or
    none), or a different lookup path (pasted text vs. search), than the
    one being asked about now - the existing "Refresh" escape hatch in
    section 6 (force a new lookup) is the intended fix for this, same as
    it is for any other stale cache entry.

    pasted_text selects Path 1; omitting it selects Path 2 (Tavily search).
    v1 treats these as mutually exclusive per call - combining facts from
    both paths in one lookup isn't attempted here.
    """
    normalized_key = normalize_entity_key(display_name)
    existing = await db.scalar(
        select(CredibilityEntity).where(CredibilityEntity.normalized_key == normalized_key)
    )
    if existing is not None:
        return existing, True

    if pasted_text:
        facts = await extract_facts_from_text(
            client, entity_type, display_name, pasted_text, tradition
        )
        fact_provenance = {key: "admin_text" for key, value in facts.items() if value is not None}
    else:
        query = _build_search_query(entity_type, display_name)
        results = await search_tavily(client, query)
        facts, fact_provenance = await extract_facts_from_search_results(
            client, entity_type, display_name, results, tradition
        )
    suggested_values = generate_suggested_values(entity_type, facts, tradition)

    entity = CredibilityEntity(
        entity_type=entity_type,
        display_name=display_name,
        normalized_key=normalized_key,
        facts=facts,
        fact_provenance=fact_provenance,
        suggested_values=suggested_values,
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return entity, False
