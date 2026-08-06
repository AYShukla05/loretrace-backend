import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.self_hosted import SelfHostedError, call_self_hosted


def run(coro):
    return asyncio.run(coro)


def test_call_self_hosted_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "self_hosted_url", None)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with pytest.raises(SelfHostedError):
        run(
            call_self_hosted(
                client, [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
            )
        )


def test_call_self_hosted_sends_auth_header_and_returns_content(monkeypatch):
    monkeypatch.setattr(settings, "self_hosted_url", "http://test-vm.example:8000")
    monkeypatch.setattr(settings, "self_hosted_api_token", "test-self-hosted-token")
    monkeypatch.setattr(settings, "self_hosted_model", "llama3.1:8b")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "self-hosted answer"}}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = run(
        call_self_hosted(
            client,
            [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "the question"},
            ],
        )
    )

    assert result == "self-hosted answer"
    assert seen["url"] == "http://test-vm.example:8000/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-self-hosted-token"
    assert seen["body"]["model"] == "llama3.1:8b"
    assert seen["body"]["messages"][1]["content"] == "the question"
