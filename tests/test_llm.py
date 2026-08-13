import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.llm import (
    SYSTEM_PROMPT,
    LLMError,
    _format_context,
    generate_answer,
    generate_stock_answer,
)
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


def test_system_prompt_requires_naming_provenance_in_prose():
    assert "name that provenance in your sentence too" in SYSTEM_PROMPT


def test_format_context_includes_source_and_tradition_labels():
    context = _format_context([make_chunk()])

    assert "https://example.com/iliad" in context
    assert "(greek)" in context
    assert "Sing, goddess, the anger of Achilles." in context


def test_format_context_omits_tradition_parenthetical_when_absent():
    chunk = make_chunk(tradition=None)

    context = _format_context([chunk])

    assert "https://example.com/iliad]" in context
    assert "(None)" not in context
    assert "()" not in context


def test_format_context_includes_provenance_tags_when_present():
    chunk = make_chunk(
        author_position=AuthorPosition.MISSIONARY,
        text_role=TextRole.SECONDARY_COMMENTARY,
        era=Era.COLONIAL_ERA,
        known_bias_flags="imposes monotheistic framing",
    )

    context = _format_context([chunk])

    assert "Provenance: missionary, secondary commentary, colonial era" in context
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


def test_generate_answer_falls_back_to_8b_on_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == settings.groq_model:
            return httpx.Response(429, json={"error": "rate limit exceeded"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "8b answer"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert result == "8b answer"
    assert calls == [settings.groq_model, settings.groq_fallback_model]


def test_generate_answer_does_not_fall_back_on_non_rate_limit_error(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert calls == [settings.groq_model]


def test_generate_answer_with_explicit_model_skips_fallback(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        run(
            generate_answer(
                client, "Who is Achilles?", [make_chunk()], model=settings.groq_fallback_model
            )
        )

    assert calls == [settings.groq_fallback_model]


def test_generate_answer_falls_back_to_cloudflare_when_8b_also_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "cloudflare_account_id", "test-account")
    monkeypatch.setattr(settings, "cloudflare_api_token", "test-cf-token")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "cloudflare.com" in str(request.url):
            calls.append("cloudflare")
            assert request.headers.get("authorization") == "Bearer test-cf-token"
            return httpx.Response(200, json={"result": {"response": "cloudflare answer"}})
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert result == "cloudflare answer"
    assert calls == [settings.groq_model, settings.groq_fallback_model, "cloudflare"]


def test_generate_answer_raises_when_all_tiers_rate_limited_and_no_cloudflare_configured(
    monkeypatch,
):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "cloudflare_account_id", None)
    monkeypatch.setattr(settings, "cloudflare_api_token", None)
    monkeypatch.setattr(settings, "self_hosted_url", None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert calls == [settings.groq_model, settings.groq_fallback_model]


def test_generate_answer_falls_back_to_self_hosted_when_cloudflare_also_rate_limited(
    monkeypatch,
):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "cloudflare_account_id", "test-account")
    monkeypatch.setattr(settings, "cloudflare_api_token", "test-cf-token")
    monkeypatch.setattr(settings, "self_hosted_url", "http://test-vm.example:8000")
    monkeypatch.setattr(settings, "self_hosted_api_token", "test-self-hosted-token")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "cloudflare.com" in url:
            calls.append("cloudflare")
            return httpx.Response(429, json={"error": "rate limit exceeded"})
        if "test-vm.example" in url:
            calls.append("self-hosted")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "self-hosted answer"}}]}
            )
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert result == "self-hosted answer"
    assert calls == [settings.groq_model, settings.groq_fallback_model, "cloudflare", "self-hosted"]


def test_generate_stock_answer_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(LLMError):
        run(generate_stock_answer(client, "Was Thor inspired by Zeus?"))


def test_generate_stock_answer_sends_no_system_prompt(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["model"] = body["model"]
        seen["messages"] = body["messages"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Thor and Zeus are both sky gods."}}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(generate_stock_answer(client, "Was Thor inspired by Zeus?"))

    assert result == "Thor and Zeus are both sky gods."
    assert seen["model"] == settings.groq_model
    assert seen["messages"] == [{"role": "user", "content": "Was Thor inspired by Zeus?"}]


def test_generate_stock_answer_raises_llm_error_on_rate_limit_without_falling_back(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMError):
        run(generate_stock_answer(client, "Was Thor inspired by Zeus?"))

    assert calls == [settings.groq_model]


def test_generate_answer_raises_when_all_four_tiers_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "cloudflare_account_id", "test-account")
    monkeypatch.setattr(settings, "cloudflare_api_token", "test-cf-token")
    monkeypatch.setattr(settings, "self_hosted_url", "http://test-vm.example:8000")
    monkeypatch.setattr(settings, "self_hosted_api_token", "test-self-hosted-token")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "cloudflare.com" in url:
            calls.append("cloudflare")
        elif "test-vm.example" in url:
            calls.append("self-hosted")
        else:
            calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        run(generate_answer(client, "Who is Achilles?", [make_chunk()]))

    assert calls == [settings.groq_model, settings.groq_fallback_model, "cloudflare", "self-hosted"]
