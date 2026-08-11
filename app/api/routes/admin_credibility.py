import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin
from app.credibility import ExtractionError, get_or_create_credibility_entity
from app.db.session import get_db
from app.models.admin import Admin
from app.schemas.credibility import CredibilityEntityRead, CredibilityLookupRequest

router = APIRouter(prefix="/admin/credibility", tags=["admin"])


@router.post("", response_model=CredibilityEntityRead)
async def lookup_credibility(
    payload: CredibilityLookupRequest,
    admin: Admin = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
) -> CredibilityEntityRead:
    async with httpx.AsyncClient() as client:
        try:
            entity, cached = await get_or_create_credibility_entity(
                db,
                client,
                payload.entity_type,
                payload.display_name,
                payload.pasted_text,
                payload.tradition,
            )
        except ExtractionError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None

    return CredibilityEntityRead(
        id=entity.id,
        entity_type=entity.entity_type,
        display_name=entity.display_name,
        normalized_key=entity.normalized_key,
        facts=entity.facts,
        fact_provenance=entity.fact_provenance,
        suggested_values=entity.suggested_values,
        suggestion_status=entity.suggestion_status,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        cached=cached,
    )
