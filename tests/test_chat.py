import asyncio

from app.api.routes import chat as chat_module
from app.retrieval import RetrievedChunk
from app.schemas.chat import ChatRequest


def run(coro):
    return asyncio.run(coro)


def make_chunk(source_id: int, source_url: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=source_id,
        source_id=source_id,
        source_url=source_url,
        tradition="greek",
        chunk_text="Zeus is the king of the gods.",
        distance=0.1,
    )


def test_chat_refuses_without_calling_llm_when_retrieval_is_empty(monkeypatch):
    async def empty_retrieve(*args, **kwargs):
        return []

    def fail_generate(*args, **kwargs):
        raise AssertionError("LLM should not be called when retrieval is empty")

    monkeypatch.setattr(chat_module, "retrieve_chunks", empty_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fail_generate)

    response = run(chat_module.chat(ChatRequest(question="Who is Loki's mother?"), db=None))

    assert response.refused is True
    assert response.answer == chat_module.REFUSAL_MESSAGE
    assert response.sources == []


def test_chat_returns_answer_and_deduped_sources_on_successful_retrieval(monkeypatch):
    chunks = [
        make_chunk(1, "https://example.com/a"),
        make_chunk(1, "https://example.com/a"),
        make_chunk(2, "https://example.com/b"),
    ]

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "Zeus is the king of the gods [Source 1]."

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

    response = run(chat_module.chat(ChatRequest(question="Who is Zeus?"), db=None))

    assert response.refused is False
    assert response.answer == "Zeus is the king of the gods [Source 1]."
    assert [source.source_id for source in response.sources] == [1, 2]
