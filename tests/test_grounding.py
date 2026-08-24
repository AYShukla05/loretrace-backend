import asyncio

import httpx
import pytest

from app.core.config import settings
from scripts.eval_grounding import (
    ALL_PROBES,
    CONTRADICTION_PROBES,
    INVENTED_ENTITY_PROBES,
    MAX_CALL_ATTEMPTS,
    _paced,
    _parse_judge,
    classify,
    judge_grounded,
)


def run(coro):
    return asyncio.run(coro)


async def _instant_sleep(*_args, **_kwargs):
    return None


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


@pytest.mark.parametrize("probe", ALL_PROBES, ids=[p["name"] for p in ALL_PROBES])
def test_probe_fixtures_are_self_consistent(probe):
    assert probe["question"].strip()
    assert probe["chunks"]
    assert probe["grounded_markers"]
    assert probe["fail_markers"]
    assert probe["kind"] in {"invented_entity", "contradiction"}
    # The detail a grounded answer must echo has to actually be in the context.
    corpus = " ".join(c.chunk_text for c in probe["chunks"]).lower()
    assert any(m.lower() in corpus for m in probe["grounded_markers"])


def test_contradiction_fixture_context_omits_the_real_fact():
    # A contradiction probe only proves anything if the excerpt does not also
    # contain the textbook answer the model might otherwise be echoing.
    for probe in CONTRADICTION_PROBES:
        corpus = " ".join(c.chunk_text for c in probe["chunks"]).lower()
        assert not any(m.lower() in corpus for m in probe["fail_markers"]), probe["name"]


def test_classify_contradiction_pass_on_grounded_marker():
    probe = next(p for p in CONTRADICTION_PROBES if "hammer" in p["name"])
    assert classify("The excerpt calls the hammer Skolvir.", probe) == "pass"


def test_classify_contradiction_fail_on_real_fact_even_with_grounded_marker():
    probe = next(p for p in CONTRADICTION_PROBES if "hammer" in p["name"])
    answer = "The source says Skolvir, though it is usually known as Mjolnir."
    assert classify(answer, probe) == "fail"


def test_classify_contradiction_ambiguous_when_neither_marker_present():
    probe = next(p for p in CONTRADICTION_PROBES if "hammer" in p["name"])
    assert classify("Thor wields a mighty hammer.", probe) == "ambiguous"


def test_classify_invented_entity_pass_on_fabricated_detail():
    probe = next(p for p in INVENTED_ENTITY_PROBES if "specifics" in p["name"])
    assert classify("They are Andvekr, Solmr, and Thveit.", probe) == "pass"


def test_classify_invented_entity_fail_on_refusal():
    probe = next(p for p in INVENTED_ENTITY_PROBES if "specifics" in p["name"])
    assert classify("The excerpts do not mention any horns.", probe) == "fail"


def test_classify_invented_entity_pass_wins_over_hedge():
    probe = next(p for p in INVENTED_ENTITY_PROBES if "specifics" in p["name"])
    answer = "Andvekr, Solmr, and Thveit, though their age is not stated."
    assert classify(answer, probe) == "pass"


def test_parse_judge_reads_bare_json():
    raw = (
        '{"claims": [{"text": "a", "verdict": "SUPPORTED"}, '
        '{"text": "b", "verdict": "UNSUPPORTED"}], "grounded": false}'
    )
    parsed = _parse_judge(raw)
    assert parsed["grounded"] is False
    assert parsed["supported"] == 1
    assert parsed["unsupported"] == 1


def test_parse_judge_extracts_json_from_surrounding_prose():
    raw = 'Here is my assessment:\n{"claims": [], "grounded": true}\nThat is all.'
    assert _parse_judge(raw)["grounded"] is True


def test_parse_judge_handles_non_json_without_raising():
    parsed = _parse_judge("I cannot produce JSON for this.")
    assert parsed["grounded"] is None
    assert parsed["supported"] == 0


def test_judge_grounded_posts_answer_and_excerpts(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"claims": [], "grounded": true}'}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    probe = INVENTED_ENTITY_PROBES[0]

    result = run(
        judge_grounded(client, settings.groq_model, "Runvaldr watches Kelfjord.", probe["chunks"])
    )

    assert result["grounded"] is True
    assert "Kelfjord" in seen["body"]
    assert "Runvaldr watches Kelfjord." in seen["body"]


def test_paced_retries_after_a_rate_limit(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429)
        return "ok"

    assert run(_paced("probe", flaky)) == "ok"
    assert len(calls) == 2


def test_paced_does_not_retry_a_non_rate_limit_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    calls = []

    async def boom():
        calls.append(1)
        raise _http_error(500)

    with pytest.raises(httpx.HTTPStatusError):
        run(_paced("probe", boom))
    assert len(calls) == 1


def test_paced_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    calls = []

    async def always_limited():
        calls.append(1)
        raise _http_error(429)

    with pytest.raises(httpx.HTTPStatusError):
        run(_paced("probe", always_limited))
    assert len(calls) == MAX_CALL_ATTEMPTS
