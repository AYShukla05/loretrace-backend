import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user_optional
from app.db.session import get_db
from app.llm import LLMError, generate_answer, generate_stock_answer
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.retrieval import RetrievedChunk, list_traditions, retrieve_chunks
from app.schemas.chat import ChatRequest, ChatResponse, CitedSource, CompareResponse

router = APIRouter(prefix="/chat", tags=["chat"])

# An empty retrieval result below RELEVANCE_THRESHOLD *is* the refusal signal
# per LoreTrace_Bias_Mitigation_Plan.md Part 1: no LLM call happens, so this
# path can't be talked around by a clever prompt.
REFUSAL_MESSAGE = "The corpus doesn't have any sources relevant enough to answer this question."


def _build_cited_sources(chunks: list[RetrievedChunk]) -> list[CitedSource]:
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
    return sources


async def _generate_grounded_response(
    client: httpx.AsyncClient, db: AsyncSession, payload: ChatRequest
) -> ChatResponse:
    chunks = await retrieve_chunks(db, payload.question, tradition=payload.tradition)
    if not chunks:
        return ChatResponse(answer=REFUSAL_MESSAGE, refused=True, sources=[])

    try:
        answer = await generate_answer(client, payload.question, chunks)
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from None

    return ChatResponse(answer=answer, refused=False, sources=_build_cited_sources(chunks))


# Anonymous chat is never saved (see LoreTrace's account-scoping decision:
# history only accrues once logged in). A conversation_id lets a logged-in
# user continue an existing thread; omitted, a new one is started.
async def _persist_message(
    db: AsyncSession, user: User, payload: ChatRequest, response: ChatResponse
) -> int:
    if payload.conversation_id is not None:
        conversation = await db.get(Conversation, payload.conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    else:
        conversation = Conversation(user_id=user.id)
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

    message = Message(
        conversation_id=conversation.id,
        question=payload.question,
        answer=response.answer,
        refused=response.refused,
        cited_source_ids=[source.source_id for source in response.sources],
    )
    db.add(message)
    await db.commit()
    return conversation.id


@router.get("/traditions", response_model=list[str])
async def traditions(db: AsyncSession = Depends(get_db)) -> list[str]:
    return await list_traditions(db)


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        response = await _generate_grounded_response(client, db, payload)

    if user is not None:
        conversation_id = await _persist_message(db, user, payload, response)
        response = response.model_copy(update={"conversation_id": conversation_id})

    return response


@router.post("/compare", response_model=CompareResponse)
async def compare(payload: ChatRequest, db: AsyncSession = Depends(get_db)) -> CompareResponse:
    async with httpx.AsyncClient() as client:
        grounded = await _generate_grounded_response(client, db, payload)

        # Deliberately caught here rather than propagated: a stock-model
        # failure shouldn't take down the grounded half of the comparison,
        # per LoreTrace_Bias_Mitigation_Plan.md Part 5 scoping (2026-08-13).
        try:
            stock_answer = await generate_stock_answer(client, payload.question)
            stock_error = None
        except LLMError as exc:
            stock_answer = None
            stock_error = str(exc)

    return CompareResponse(
        question=payload.question,
        stock_answer=stock_answer,
        stock_error=stock_error,
        grounded=grounded,
    )
