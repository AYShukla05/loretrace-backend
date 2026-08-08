from pydantic import BaseModel

from app.models.enums import AuthorPosition


class ChatRequest(BaseModel):
    question: str
    tradition: str | None = None


class CitedSource(BaseModel):
    source_id: int
    source_url: str
    tradition: str | None
    author_position: AuthorPosition | None


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[CitedSource]
