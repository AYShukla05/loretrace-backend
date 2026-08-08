from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from app.models.enums import AuthorPosition, Era, SourceStatus, SourceType, TextRole


class SourceCreate(BaseModel):
    url: HttpUrl
    source_type: SourceType | None = None
    tradition: str | None = Field(default=None, max_length=50)
    era: Era | None = None
    author_position: AuthorPosition | None = None
    text_role: TextRole | None = None
    known_bias_flags: str | None = None


class SourceRead(BaseModel):
    id: int
    url: str
    source_type: SourceType | None
    tradition: str | None
    status: SourceStatus
    era: Era | None
    author_position: AuthorPosition | None
    text_role: TextRole | None
    known_bias_flags: str | None
    last_scraped_at: datetime | None
    last_checked_at: datetime | None
    created_at: datetime
    chunk_count: int
