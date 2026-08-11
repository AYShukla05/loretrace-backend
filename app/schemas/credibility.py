from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import CredibilityEntityType, SuggestionStatus


class CredibilityLookupRequest(BaseModel):
    entity_type: CredibilityEntityType
    display_name: str = Field(min_length=1, max_length=255)
    # Path 1 (admin-pasted text) when set; omit for Path 2 (Tavily search)
    # instead - see app/credibility.py::get_or_create_credibility_entity.
    pasted_text: str | None = Field(default=None, min_length=1)
    # The source's tradition (Source.tradition), if the lookup is happening
    # from a specific source's form. author_origin is only suggested when
    # this is present - see app/credibility.py::generate_suggested_values.
    tradition: str | None = Field(default=None, max_length=50)


class CredibilityEntityRead(BaseModel):
    id: int
    entity_type: CredibilityEntityType
    display_name: str
    normalized_key: str
    facts: dict[str, Any]
    fact_provenance: dict[str, Any]
    suggested_values: dict[str, Any]
    suggestion_status: SuggestionStatus
    created_at: datetime
    updated_at: datetime
    cached: bool
