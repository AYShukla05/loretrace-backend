import asyncio

import httpx
import pytest

from app.core.config import settings
from app.llm import SYSTEM_PROMPT, LLMError, _format_context, generate_answer
from app.models.enums import AuthorPosition, Era, TextRole
from app.retrieval import RetrievedChunk


def run(coro):
    return asyncio.run(coro)


def make_chunk(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=1,
        source_id=1,
        source_url="https://example.com/iliad",
        tradition="greek",
        chunk_text="Sing, goddess, the anger of Achilles.",
        distance=0.2,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_system_prompt_forbids_cross_source_synthesis():
    assert "cross-source synthesis" in SYSTEM_PROMPT


def test_system_prompt_forbids_unsourced_hedging():
    assert "unsourced hedging" in SYSTEM_PROMPT


def test_system_prompt_requires_attributed_disagreement():
    assert "present each one separately" in SYSTEM_PROMPT


def test_format_context_includes_source_and_tradition_labels():
    context = _format_context([make_chunk()])

    assert "https://example.com/iliad" in context
    assert "(greek)" in context
    assert "Sing, goddess, the anger of Achilles." in context


def test_format_context_includes_provenance_tags_when_present():
    chunk = make_chunk(
        author_position=AuthorPosition.MISSIONARY,
        text_role=TextRole.SECONDARY_COMMENTARY,
        era=Era.COLONIAL_ERA,
        known_bias_flags="imposes monotheistic framing",
    )

    context = _format_context([chunk])

    assert "missionary" in context
    assert "secondary commentary" in context
    assert "colonial era" in context
    assert "flagged: imposes monotheistic framing" in context


def test_format_context_omits_provenance_tags_when_absent():
    context = _format_context([make_chunk()])

    assert context.count("[Source 1:") == 1
    assert "flagged:" not in context


def test_generate_answer_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(LLMError):
        run(generate_answer(client, "Who is Achilles?", [make_chunk()]))


def test_generate_answer_sends_auth_header_and_returns_content(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Achilles is the son of Peleus and Thetis."}}]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert result == "Achilles is the son of Peleus and Thetis."
    assert seen["authorization"] == "Bearer test-key"
    assert "Sing, goddess, the anger of Achilles." in seen["body"]
