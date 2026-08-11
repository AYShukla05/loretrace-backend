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
    pasted_text: str,
    tradition: str | None = None,
) -> tuple[CredibilityEntity, bool]:
    """Cache-first per section 6: an existing row for this normalized name is
    returned as-is, with no extraction call. Returns (entity, cache_hit).

    Known limitation, not addressed by the original design doc: a cache hit
    returns whatever suggested_values were generated the first time this
    entity was looked up, which may have used a different tradition (or
    none) than the one being asked about now - tradition-specific facts
    like tradition_engagement aren't re-checked on a hit. The existing
    "Refresh" escape hatch in section 6 (force a new lookup) is the
    intended fix for this, same as it is for any other stale cache entry.
    """
    normalized_key = normalize_entity_key(display_name)
    existing = await db.scalar(
        select(CredibilityEntity).where(CredibilityEntity.normalized_key == normalized_key)
    )
    if existing is not None:
        return existing, True

    facts = await extract_facts_from_text(client, entity_type, display_name, pasted_text, tradition)
    fact_provenance = {key: "admin_text" for key, value in facts.items() if value is not None}
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
