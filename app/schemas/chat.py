from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    tradition: str | None = None


class CitedSource(BaseModel):
    source_id: int
    source_url: str
    tradition: str


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[CitedSource]
