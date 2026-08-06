import httpx

from app.core.config import settings


class SelfHostedError(RuntimeError):
    pass


async def call_self_hosted(client: httpx.AsyncClient, messages: list[dict]) -> str:
    if not settings.self_hosted_url:
        raise SelfHostedError("SELF_HOSTED_URL is not configured")
    response = await client.post(
        f"{settings.self_hosted_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.self_hosted_api_token}"},
        json={
            "model": settings.self_hosted_model,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
