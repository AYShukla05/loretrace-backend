import asyncio

import pytest

from app.core.config import settings
from app.self_hosted import SelfHostedError, call_self_hosted


def run(coro):
    return asyncio.run(coro)


def test_call_self_hosted_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "self_hosted_space", None)

    with pytest.raises(SelfHostedError):
        run(
            call_self_hosted(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
            )
        )


def test_call_self_hosted_calls_predict_with_system_and_user_messages(monkeypatch):
    monkeypatch.setattr(settings, "self_hosted_space", "test-user/test-space")
    monkeypatch.setattr(settings, "self_hosted_api_name", "/predict")
    monkeypatch.setattr(settings, "self_hosted_hf_token", "test-hf-token")
    seen = {}

    class FakeClient:
        def __init__(self, src, token=None, verbose=True):
            seen["src"] = src
            seen["token"] = token
            seen["verbose"] = verbose

        def predict(self, *args, api_name=None):
            seen["args"] = args
            seen["api_name"] = api_name
            return "self-hosted answer"

    monkeypatch.setattr("app.self_hosted.Client", FakeClient)

    result = run(
        call_self_hosted(
            [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "the question"},
            ]
        )
    )

    assert result == "self-hosted answer"
    assert seen["src"] == "test-user/test-space"
    assert seen["token"] == "test-hf-token"
    assert seen["verbose"] is False
    assert seen["args"] == ("sys prompt", "the question")
    assert seen["api_name"] == "/predict"
