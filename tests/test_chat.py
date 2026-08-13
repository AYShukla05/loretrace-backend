import asyncio

from app.api.routes import chat as chat_module
from app.llm import LLMError
from app.models.enums import AuthorPosition
from app.retrieval import RetrievedChunk
from app.schemas.chat import ChatRequest


def run(coro):
    return asyncio.run(coro)


def make_chunk(
    source_id: int, source_url: str, author_position: AuthorPosition | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=source_id,
        source_id=source_id,
        source_url=source_url,
        tradition="greek",
        chunk_text="Zeus is the king of the gods.",
        distance=0.1,
        author_position=author_position,
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


def test_chat_carries_author_position_onto_cited_sources(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a", AuthorPosition.INDIGENOUS_PRIMARY_TEXT)]

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "answer"

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

    response = run(chat_module.chat(ChatRequest(question="Who is Zeus?"), db=None))

    assert response.sources[0].author_position == AuthorPosition.INDIGENOUS_PRIMARY_TEXT


def test_traditions_route_returns_list_traditions_result(monkeypatch):
    async def fake_list_traditions(db):
        return ["greek", "norse"]

    monkeypatch.setattr(chat_module, "list_traditions", fake_list_traditions)

    result = run(chat_module.traditions(db=None))

    assert result == ["greek", "norse"]


def test_compare_returns_both_stock_and_grounded_answers_on_success(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a")]

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "Zeus is the king of the gods [Source 1]."

    async def fake_generate_stock(client, query):
        return "Thor was likely inspired by Zeus."

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "generate_stock_answer", fake_generate_stock)

    response = run(chat_module.compare(ChatRequest(question="Was Thor inspired by Zeus?"), db=None))

    assert response.question == "Was Thor inspired by Zeus?"
    assert response.stock_answer == "Thor was likely inspired by Zeus."
    assert response.stock_error is None
    assert response.grounded.refused is False
    assert response.grounded.answer == "Zeus is the king of the gods [Source 1]."


def test_compare_still_attempts_stock_call_when_grounded_side_refuses(monkeypatch):
    async def empty_retrieve(*args, **kwargs):
        return []

    async def fake_generate_stock(client, query):
        return "Ragnarok is the Norse apocalypse."

    monkeypatch.setattr(chat_module, "retrieve_chunks", empty_retrieve)
    monkeypatch.setattr(chat_module, "generate_stock_answer", fake_generate_stock)

    response = run(chat_module.compare(ChatRequest(question="What is Ragnarok?"), db=None))

    assert response.grounded.refused is True
    assert response.grounded.answer == chat_module.REFUSAL_MESSAGE
    assert response.stock_answer == "Ragnarok is the Norse apocalypse."
    assert response.stock_error is None


def test_compare_reports_stock_error_without_failing_grounded_side(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a")]

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "Zeus is the king of the gods [Source 1]."

    async def failing_generate_stock(client, query):
        raise LLMError("stock model request failed: 429")

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)
    monkeypatch.setattr(chat_module, "generate_stock_answer", failing_generate_stock)

    response = run(chat_module.compare(ChatRequest(question="Who is Zeus?"), db=None))

    assert response.grounded.refused is False
    assert response.grounded.answer == "Zeus is the king of the gods [Source 1]."
    assert response.stock_answer is None
    assert response.stock_error == "stock model request failed: 429"
