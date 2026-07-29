from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from app.models.enums import SourceStatus, SourceType


class SourceCreate(BaseModel):
    url: HttpUrl
    source_type: SourceType
    tradition: str = Field(max_length=50)


class SourceRead(BaseModel):
    id: int
    url: str
    source_type: SourceType
    tradition: str
    status: SourceStatus
    last_scraped_at: datetime | None
    last_checked_at: datetime | None
    created_at: datetime
    chunk_count: int
