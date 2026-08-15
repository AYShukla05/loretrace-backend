from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.models.enums import (
    AuthorEpistemicBasis,
    AuthorOrigin,
    AuthorPosition,
    Era,
    HistoriographicalMethod,
    SourceStatus,
    SourceType,
    TextRole,
)


def normalize_tradition(value: str | None) -> str | None:
    """Collapse whitespace/casing variants ("norse", " Norse ", "NORSE") into
    one canonical form, so the same tradition doesn't fragment into several
    near-duplicate values across sources entered by different admins."""
    if value is None:
        return None
    collapsed = " ".join(value.split())
    return collapsed.title() if collapsed else None


class SourceCreate(BaseModel):
    url: HttpUrl
    # Normally left unset and filled in automatically during scraping; an
    # admin can still set it directly here to skip or pre-empt inference.
    title: str | None = Field(default=None, max_length=255)
    source_type: SourceType | None = None
    tradition: str | None = Field(default=None, max_length=50)
    era: Era | None = None
    author_position: AuthorPosition | None = None
    text_role: TextRole | None = None
    known_bias_flags: str | None = None
    historiographical_method: HistoriographicalMethod | None = None
    author_origin: AuthorOrigin | None = None
    author_epistemic_basis: AuthorEpistemicBasis | None = None

    _normalize_tradition = field_validator("tradition")(normalize_tradition)


class SourceUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    source_type: SourceType | None = None
    tradition: str | None = Field(default=None, max_length=50)
    era: Era | None = None
    author_position: AuthorPosition | None = None
    text_role: TextRole | None = None
    known_bias_flags: str | None = None
    historiographical_method: HistoriographicalMethod | None = None
    author_origin: AuthorOrigin | None = None
    author_epistemic_basis: AuthorEpistemicBasis | None = None

    _normalize_tradition = field_validator("tradition")(normalize_tradition)


class SourceRead(BaseModel):
    id: int
    url: str
    title: str | None
    source_type: SourceType | None
    tradition: str | None
    status: SourceStatus
    era: Era | None
    author_position: AuthorPosition | None
    text_role: TextRole | None
    known_bias_flags: str | None
    historiographical_method: HistoriographicalMethod | None
    author_origin: AuthorOrigin | None
    author_epistemic_basis: AuthorEpistemicBasis | None
    last_scraped_at: datetime | None
    last_checked_at: datetime | None
    created_at: datetime
    chunk_count: int
