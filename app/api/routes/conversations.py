from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.source import Source
from app.models.user import User
from app.schemas.chat import CitedSource
from app.schemas.conversation import ConversationDetail, ConversationSummary, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])


async def _hydrate_sources(db: AsyncSession, messages: list[Message]) -> dict[int, CitedSource]:
    source_ids = {source_id for message in messages for source_id in message.cited_source_ids}
    if not source_ids:
        return {}

    sources = await db.scalars(select(Source).where(Source.id.in_(source_ids)))
    return {
        source.id: CitedSource(
            source_id=source.id,
            source_url=source.url,
            tradition=source.tradition,
            author_position=source.author_position,
        )
        for source in sources
    }


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ConversationSummary]:
    conversations = list(
        await db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.created_at.desc())
        )
    )
    if not conversations:
        return []

    # Two queries total regardless of conversation count, not one per
    # conversation. Messages come back newest-first, so the first one
    # appended per conversation_id is that conversation's most recent.
    conversation_ids = [conversation.id for conversation in conversations]
    messages = await db.scalars(
        select(Message)
        .where(Message.conversation_id.in_(conversation_ids))
        .order_by(Message.created_at.desc())
    )
    by_conversation: dict[int, list[Message]] = defaultdict(list)
    for message in messages:
        by_conversation[message.conversation_id].append(message)

    return [
        ConversationSummary(
            id=conversation.id,
            created_at=conversation.created_at,
            message_count=len(by_conversation[conversation.id]),
            last_question=(
                by_conversation[conversation.id][0].question
                if by_conversation[conversation.id]
                else None
            ),
        )
        for conversation in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationDetail:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = list(
        await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
    )
    hydrated_sources = await _hydrate_sources(db, messages)

    return ConversationDetail(
        id=conversation.id,
        created_at=conversation.created_at,
        messages=[
            MessageRead(
                id=message.id,
                question=message.question,
                answer=message.answer,
                refused=message.refused,
                sources=[
                    hydrated_sources[source_id]
                    for source_id in message.cited_source_ids
                    if source_id in hydrated_sources
                ],
                created_at=message.created_at,
            )
            for message in messages
        ],
    )
