from datetime import datetime

from pydantic import BaseModel

from app.schemas.chat import CitedSource


class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    message_count: int
    last_question: str | None


class MessageRead(BaseModel):
    id: int
    question: str
    answer: str
    refused: bool
    sources: list[CitedSource]
    created_at: datetime


class ConversationDetail(BaseModel):
    id: int
    created_at: datetime
    messages: list[MessageRead]
