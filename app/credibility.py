import json
from typing import Any

import httpx

from app.core.config import settings
from app.models.enums import CredibilityEntityType

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Fields an admin might plausibly document for an author or institution. Not
# every field applies to every entity_type (e.g. practice_lineage rarely
# applies to an institution) - the model is told to use null rather than
# guess when a field isn't covered by the pasted text.
FACT_KEYS = (
    "birth_year",
    "death_year",
    "residence_regions",
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
"birth_year" and "death_year" are integers or null. "residence_regions", \
"institutional_affiliations", and "documented_critique_refs" are arrays of strings, or null. \
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
