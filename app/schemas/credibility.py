from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import CredibilityEntityType, SuggestionStatus


class CredibilityLookupRequest(BaseModel):
    entity_type: CredibilityEntityType
    display_name: str = Field(min_length=1, max_length=255)
    pasted_text: str = Field(min_length=1)


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
