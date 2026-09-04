from pydantic import BaseModel

from app.models.enums import AuthorPosition


class ChatRequest(BaseModel):
    question: str
    tradition: str | None = None
    # Set to continue an existing thread (must belong to the authenticated
    # user); omitted starts a new one. Ignored entirely for anonymous chat.
    conversation_id: int | None = None


class CitedSource(BaseModel):
    source_id: int
    source_url: str
    tradition: str | None
    author_position: AuthorPosition | None
    title: str | None
    # The specific work cited within a multi-work volume, when the source is
    # one. None for a single-work source (cite by title). Saved chat history
    # only records source ids, so a rebuilt historical citation leaves this
    # null even if the live answer had it.
    work_title: str | None = None


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[CitedSource]
    # Set only when the exchange was actually persisted (a logged-in user);
    # null for anonymous chat, which is never saved.
    conversation_id: int | None = None


class CompareResponse(BaseModel):
    question: str
    stock_answer: str | None
    stock_error: str | None
    grounded: ChatResponse


class CorpusText(BaseModel):
    title: str | None
    url: str


class CorpusTradition(BaseModel):
    tradition: str
    texts: list[CorpusText]


class CorpusOverview(BaseModel):
    traditions: list[CorpusTradition]
    text_count: int
