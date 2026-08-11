import json
import re
import unicodedata
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.credibility_entity import CredibilityEntity
from app.models.enums import CredibilityEntityType

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

# Path 1 of LoreTrace_Credibility_Suggestion_Design.md section 5: the model is
# used in a strictly extractive role. It must never add anything from its own
# training knowledge about the entity, only what the pasted text states -
# this is the same "no facts beyond what's given" discipline app/llm.py's
# SYSTEM_PROMPT enforces on chat answers, applied one layer earlier.
EXTRACTION_SYSTEM_PROMPT = f"""You extract structured facts about a historical or contemporary \
person or institution from a piece of text an admin has pasted. Follow these rules exactly:

1. Use only facts stated in the pasted text. Never add anything from your own training knowledge \
about the named entity, even if you recognize them and know more.
2. If the text doesn't state a fact, its value is null. Do not guess or infer beyond what the text \
says.
3. Never assert a verdict on a contested claim (for example, whether a text or tradition is \
historically accurate). Only extract biographical and institutional facts.
4. Respond with a single JSON object with exactly these keys: {", ".join(FACT_KEYS)}. \
"birth_year" and "death_year" are integers or null. "birth_region" is the region/country the \
person was born or raised in, as a string or null. "practice_regions" is where the person lived, \
worked, or practiced later in life, as an array of strings or null - this can differ from \
birth_region and both should reflect only what the text actually says. \
"institutional_affiliations" and "documented_critique_refs" are arrays of strings, or null. \
"occupation" and "practice_lineage" are strings or null."""


class ExtractionError(RuntimeError):
    pass


async def extract_facts_from_text(
    client: httpx.AsyncClient,
    entity_type: CredibilityEntityType,
    display_name: str,
    pasted_text: str,
) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise ExtractionError("GROQ_API_KEY is not configured")

    user_content = (
        f"Entity type: {entity_type.value}\nEntity name: {display_name}\n\nText:\n{pasted_text}"
    )
    response = await client.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            # A structured-extraction task doesn't need the 70B tier's
            # quota - reserve that for actual chat answers.
            "model": settings.groq_fallback_model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
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
    return {key: parsed.get(key) for key in FACT_KEYS}


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
) -> tuple[CredibilityEntity, bool]:
    """Cache-first per section 6: an existing row for this normalized name is
    returned as-is, with no extraction call. Returns (entity, cache_hit).

    Rule-table suggestion generation (section 7) isn't built yet -
    suggested_values stays empty until that's implemented.
    """
    normalized_key = normalize_entity_key(display_name)
    existing = await db.scalar(
        select(CredibilityEntity).where(CredibilityEntity.normalized_key == normalized_key)
    )
    if existing is not None:
        return existing, True

    facts = await extract_facts_from_text(client, entity_type, display_name, pasted_text)
    fact_provenance = {key: "admin_text" for key, value in facts.items() if value is not None}

    entity = CredibilityEntity(
        entity_type=entity_type,
        display_name=display_name,
        normalized_key=normalized_key,
        facts=facts,
        fact_provenance=fact_provenance,
        suggested_values={},
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return entity, False
