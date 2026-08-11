import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.credibility import (
    ExtractionError,
    extract_facts_from_text,
    generate_suggested_values,
    normalize_entity_key,
)
from app.models.enums import AuthorEpistemicBasis, AuthorOrigin, CredibilityEntityType


def run(coro):
    return asyncio.run(coro)


def test_extract_facts_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(ExtractionError):
        run(
            extract_facts_from_text(
                client, CredibilityEntityType.AUTHOR, "Max Muller", "Some pasted text."
            )
        )


def test_extract_facts_sends_auth_header_and_entity_details(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "birth_year": 1823,
                                    "death_year": 1900,
                                    "birth_region": "Germany",
                                    "practice_regions": ["England"],
                                    "occupation": "philologist",
                                    "institutional_affiliations": ["Oxford University"],
                                    "practice_lineage": None,
                                    "documented_critique_refs": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    facts = run(
        extract_facts_from_text(
            client, CredibilityEntityType.AUTHOR, "Max Muller", "Born in Germany in 1823..."
        )
    )

    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"]["model"] == settings.groq_fallback_model
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert "Max Muller" in seen["body"]["messages"][1]["content"]
    assert facts["birth_year"] == 1823
    assert facts["occupation"] == "philologist"
    assert facts["institutional_affiliations"] == ["Oxford University"]


def test_extract_facts_defaults_missing_keys_to_none(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"birth_year": 1900})}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    facts = run(
        extract_facts_from_text(client, CredibilityEntityType.INSTITUTION, "Some Press", "text")
    )

    assert facts["birth_year"] == 1900
    assert facts["occupation"] is None
    assert facts["documented_critique_refs"] is None


def test_extract_facts_drops_unknown_keys_from_model_output(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"birth_year": 1900, "made_up_field": "nope"})
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    facts = run(extract_facts_from_text(client, CredibilityEntityType.AUTHOR, "Someone", "text"))

    assert "made_up_field" not in facts


def test_extract_facts_requests_tradition_facts_when_tradition_given(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"birth_year": 1823, "born_within_tradition_context": False}
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    facts = run(
        extract_facts_from_text(
            client, CredibilityEntityType.AUTHOR, "Max Muller", "text", tradition="norse"
        )
    )

    assert "norse" in seen["body"]["messages"][0]["content"]
    assert "norse" in seen["body"]["messages"][1]["content"]
    assert "born_within_tradition_context" in facts
    assert facts["born_within_tradition_context"] is False
    assert "tradition_engagement" in facts


def test_extract_facts_omits_tradition_facts_without_tradition(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"birth_year": 1823})}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    facts = run(extract_facts_from_text(client, CredibilityEntityType.AUTHOR, "Max Muller", "text"))

    assert "born_within_tradition_context" not in facts
    assert "tradition_engagement" not in facts


def test_suggest_epistemic_basis_textual_study_only_when_engagement_explicitly_false():
    facts = {"practice_lineage": None, "tradition_engagement": False}
    suggested = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    assert (
        suggested["author_epistemic_basis"]["value"]
        == AuthorEpistemicBasis.TEXTUAL_STUDY_ONLY.value
    )


def test_suggest_epistemic_basis_lived_practice_when_lineage_documented():
    facts = {"practice_lineage": "trained under a named guru", "tradition_engagement": True}
    suggested = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    assert suggested["author_epistemic_basis"]["value"] == AuthorEpistemicBasis.LIVED_PRACTICE.value


def test_suggest_epistemic_basis_mixed_when_engagement_true_but_no_lineage():
    facts = {"practice_lineage": None, "tradition_engagement": True}
    suggested = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    assert suggested["author_epistemic_basis"]["value"] == AuthorEpistemicBasis.MIXED.value


def test_suggest_epistemic_basis_unknown_when_facts_thin():
    facts = {"practice_lineage": None, "tradition_engagement": None}
    suggested = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    assert suggested["author_epistemic_basis"]["value"] == AuthorEpistemicBasis.UNKNOWN.value


def test_suggest_author_origin_only_generated_when_tradition_given():
    facts = {"born_within_tradition_context": True}
    with_tradition = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    without_tradition = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, None)

    assert with_tradition["author_origin"]["value"] == AuthorOrigin.INDIGENOUS_BORN.value
    assert "author_origin" not in without_tradition


def test_suggest_author_origin_foreign_born():
    facts = {"born_within_tradition_context": False}
    suggested = generate_suggested_values(CredibilityEntityType.AUTHOR, facts, "vedic")
    assert suggested["author_origin"]["value"] == AuthorOrigin.FOREIGN_BORN.value


def test_generate_suggested_values_empty_for_institution():
    facts = {"born_within_tradition_context": True, "practice_lineage": "x"}
    suggested = generate_suggested_values(CredibilityEntityType.INSTITUTION, facts, "vedic")
    assert suggested == {}


def test_normalize_entity_key_strips_diacritics_and_punctuation():
    assert normalize_entity_key("Max Müller") == "max muller"
    assert normalize_entity_key("F. Max Müller") == "f max muller"
    assert normalize_entity_key("Max Mueller") == "max mueller"


def test_normalize_entity_key_collapses_whitespace_and_case():
    assert normalize_entity_key("  Max   MULLER  ") == "max muller"


def test_extract_facts_raises_on_invalid_json(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ExtractionError):
        run(extract_facts_from_text(client, CredibilityEntityType.AUTHOR, "Someone", "text"))
