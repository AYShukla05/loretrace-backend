import asyncio

from gradio_client import Client

from app.core.config import settings


class SelfHostedError(RuntimeError):
    pass


def _predict_sync(system_prompt: str, user_message: str) -> str:
    if not settings.self_hosted_space:
        raise SelfHostedError("SELF_HOSTED_SPACE is not configured")
    client = Client(settings.self_hosted_space, token=settings.self_hosted_hf_token, verbose=False)
    return client.predict(system_prompt, user_message, api_name=settings.self_hosted_api_name)


async def call_self_hosted(messages: list[dict]) -> str:
    system_prompt = messages[0]["content"]
    user_message = messages[-1]["content"]
    return await asyncio.to_thread(_predict_sync, system_prompt, user_message)
