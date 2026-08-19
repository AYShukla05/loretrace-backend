import asyncio

from app.api.routes import chat as chat_module
from app.llm import LLMError
from app.models.conversation import Conversation
from app.models.enums import AuthorPosition
from app.models.user import User
from app.retrieval import RetrievedChunk
from app.schemas.chat import ChatRequest


def run(coro):
    return asyncio.run(coro)


class FakeSession:
    """Records add()/commit() calls; db.get() returns a preset object.

    Mirrors the FakeSession pattern in tests/test_queue.py. commit() assigns
    a fake id to anything added that doesn't have one yet, standing in for
    the real DB's autoincrement.
    """

    def __init__(self, get_result=None):
        self.added = []
        self.commits = 0
        self._get_result = get_result
        self._next_id = 100

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_id
                self._next_id += 1

    async def refresh(self, obj):
        pass

    async def get(self, model, id_):
        return self._get_result


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

    response = run(
        chat_module.chat(ChatRequest(question="Who is Loki's mother?"), db=None, user=None)
    )

    assert response.refused is True
    assert response.answer == chat_module.REFUSAL_MESSAGE
    assert response.sources == []
    assert response.conversation_id is None


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

    response = run(chat_module.chat(ChatRequest(question="Who is Zeus?"), db=None, user=None))

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

    response = run(chat_module.chat(ChatRequest(question="Who is Zeus?"), db=None, user=None))

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


def test_chat_persists_new_conversation_when_user_is_authenticated(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a")]
    user = User(id=7, email="reader@example.com", password_hash="x")
    db = FakeSession()

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "Zeus is the king of the gods [Source 1]."

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

    response = run(chat_module.chat(ChatRequest(question="Who is Zeus?"), db=db, user=user))

    assert response.conversation_id is not None
    conversations = [obj for obj in db.added if isinstance(obj, Conversation)]
    messages = [obj for obj in db.added if isinstance(obj, chat_module.Message)]
    assert len(conversations) == 1
    assert conversations[0].user_id == 7
    assert len(messages) == 1
    assert messages[0].conversation_id == response.conversation_id
    assert messages[0].question == "Who is Zeus?"
    assert messages[0].answer == "Zeus is the king of the gods [Source 1]."
    assert messages[0].refused is False
    assert messages[0].cited_source_ids == [1]


def test_chat_persists_refusal_when_user_is_authenticated(monkeypatch):
    user = User(id=7, email="reader@example.com", password_hash="x")
    db = FakeSession()

    async def empty_retrieve(*args, **kwargs):
        return []

    monkeypatch.setattr(chat_module, "retrieve_chunks", empty_retrieve)

    response = run(
        chat_module.chat(ChatRequest(question="Who is Loki's mother?"), db=db, user=user)
    )

    assert response.conversation_id is not None
    messages = [obj for obj in db.added if isinstance(obj, chat_module.Message)]
    assert len(messages) == 1
    assert messages[0].refused is True
    assert messages[0].cited_source_ids == []


def test_chat_appends_to_existing_conversation_owned_by_the_user(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a")]
    user = User(id=7, email="reader@example.com", password_hash="x")
    existing = Conversation(id=42, user_id=7)
    db = FakeSession(get_result=existing)

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "answer"

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

    response = run(
        chat_module.chat(ChatRequest(question="And Hera?", conversation_id=42), db=db, user=user)
    )

    assert response.conversation_id == 42
    conversations = [obj for obj in db.added if isinstance(obj, Conversation)]
    assert conversations == []  # no new conversation created, only the message


def test_chat_rejects_conversation_id_owned_by_another_user(monkeypatch):
    chunks = [make_chunk(1, "https://example.com/a")]
    user = User(id=7, email="reader@example.com", password_hash="x")
    someone_elses = Conversation(id=42, user_id=99)
    db = FakeSession(get_result=someone_elses)

    async def fake_retrieve(*args, **kwargs):
        return chunks

    async def fake_generate(client, query, chunks):
        return "answer"

    monkeypatch.setattr(chat_module, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(chat_module, "generate_answer", fake_generate)

    try:
        run(
            chat_module.chat(
                ChatRequest(question="And Hera?", conversation_id=42), db=db, user=user
            )
        )
        raise AssertionError("expected a 404 for a conversation owned by another user")
    except chat_module.HTTPException as exc:
        assert exc.status_code == 404
