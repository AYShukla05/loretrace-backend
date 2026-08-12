import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm import LLMError, generate_answer
from app.retrieval import list_traditions, retrieve_chunks
from app.schemas.chat import ChatRequest, ChatResponse, CitedSource

router = APIRouter(prefix="/chat", tags=["chat"])

# An empty retrieval result below RELEVANCE_THRESHOLD *is* the refusal signal
# per LoreTrace_Bias_Mitigation_Plan.md Part 1: no LLM call happens, so this
# path can't be talked around by a clever prompt.
REFUSAL_MESSAGE = "The corpus doesn't have any sources relevant enough to answer this question."


@router.get("/traditions", response_model=list[str])
async def traditions(db: AsyncSession = Depends(get_db)) -> list[str]:
    return await list_traditions(db)


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    chunks = await retrieve_chunks(db, payload.question, tradition=payload.tradition)
    if not chunks:
        return ChatResponse(answer=REFUSAL_MESSAGE, refused=True, sources=[])

    async with httpx.AsyncClient() as client:
        try:
            answer = await generate_answer(client, payload.question, chunks)
        except LLMError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None

    seen_source_ids: set[int] = set()
    sources = []
    for chunk in chunks:
        if chunk.source_id in seen_source_ids:
            continue
        seen_source_ids.add(chunk.source_id)
        sources.append(
            CitedSource(
                source_id=chunk.source_id,
                source_url=chunk.source_url,
                tradition=chunk.tradition,
                author_position=chunk.author_position,
            )
        )

    return ChatResponse(answer=answer, refused=False, sources=sources)
