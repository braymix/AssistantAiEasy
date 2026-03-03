"""
Conversation Manager – handles conversation lifecycle and message persistence.

Provides a DI-ready service class that encapsulates all conversation-related
database operations, designed to be injected into FastAPI route handlers via
``Depends(get_conversation_manager)``.
"""

from uuid import UUID

from fastapi import Depends
from sqlalchemy import String, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.logging import get_logger
from src.shared.database import get_db_session
from src.shared.models import (
    Conversation,
    KnowledgeItem,
    Message,
    MessageRole,
)

logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation lifecycle, message persistence, and knowledge extraction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # 1. create_or_get_conversation
    # ------------------------------------------------------------------

    async def create_or_get_conversation(self, session_id: str) -> Conversation:
        """Create a new conversation or retrieve an existing one by *session_id*.

        Open WebUI sends a stable ``session_id`` per chat window.  We reuse
        the existing conversation so that the full history is kept in one row
        rather than creating a duplicate on every request.
        """
        stmt = (
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(Conversation.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        conversation = result.scalar_one_or_none()

        if conversation is not None:
            logger.info(
                "conversation_reused",
                conversation_id=conversation.id,
                session_id=session_id,
            )
            return conversation

        conversation = Conversation(session_id=session_id, metadata_json={})
        self._session.add(conversation)
        await self._session.flush()

        logger.info(
            "conversation_created",
            conversation_id=conversation.id,
            session_id=session_id,
        )
        return conversation

    # ------------------------------------------------------------------
    # 2. add_message
    # ------------------------------------------------------------------

    async def add_message(
        self,
        conversation_id: str | UUID,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> Message:
        """Persist a single message in the given conversation.

        Parameters
        ----------
        conversation_id:
            FK to ``conversations.id``.
        role:
            One of ``"user"``, ``"assistant"``, ``"system"``.
        content:
            The message body.
        metadata:
            Optional dict – if it contains a ``"detected_contexts"`` key,
            its value is stored on the message.

        Returns the newly created :class:`Message` instance.
        """
        try:
            msg_role = MessageRole(role)
        except ValueError:
            msg_role = MessageRole.USER

        message = Message(
            conversation_id=str(conversation_id),
            role=msg_role,
            content=content,
            detected_contexts=metadata.get("detected_contexts") if metadata else None,
        )
        self._session.add(message)
        await self._session.flush()

        logger.info(
            "message_added",
            message_id=message.id,
            conversation_id=str(conversation_id),
            role=role,
        )
        return message

    # ------------------------------------------------------------------
    # 3. get_conversation_history
    # ------------------------------------------------------------------

    async def get_conversation_history(
        self,
        conversation_id: str | UUID,
        limit: int = 50,
    ) -> list[Message]:
        """Return the most recent *limit* messages for a conversation,
        ordered chronologically (oldest first).
        """
        stmt = (
            select(Message)
            .where(Message.conversation_id == str(conversation_id))
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())

        logger.debug(
            "conversation_history_loaded",
            conversation_id=str(conversation_id),
            message_count=len(messages),
        )
        return messages

    # ------------------------------------------------------------------
    # 4. get_conversations_by_context
    # ------------------------------------------------------------------

    async def get_conversations_by_context(
        self,
        context_name: str,
        limit: int = 20,
    ) -> list[Conversation]:
        """Return conversations that contain messages referencing *context_name*.

        Searches ``messages.detected_contexts`` (a JSON list) for entries
        matching the requested context name.  Uses a text ``LIKE`` on the
        serialised column which works for both SQLite and PostgreSQL.
        """
        msg_subq = (
            select(Message.conversation_id)
            .where(Message.detected_contexts.isnot(None))
            .where(
                Message.detected_contexts.cast(String).contains(context_name)
            )
            .distinct()
            .subquery()
        )

        stmt = (
            select(Conversation)
            .where(Conversation.id.in_(select(msg_subq.c.conversation_id)))
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        conversations = list(result.scalars().all())

        logger.info(
            "conversations_by_context",
            context=context_name,
            found=len(conversations),
        )
        return conversations

    # ------------------------------------------------------------------
    # 5. mark_knowledge_extracted
    # ------------------------------------------------------------------

    async def mark_knowledge_extracted(
        self,
        message_id: str | UUID,
        knowledge_item_id: str | UUID,
    ) -> None:
        """Link a :class:`KnowledgeItem` to its source message and flag the
        message as having had knowledge extracted.
        """
        stmt = select(Message).where(Message.id == str(message_id))
        result = await self._session.execute(stmt)
        message = result.scalar_one_or_none()
        if message is None:
            logger.warning("mark_knowledge_extracted_message_not_found", message_id=str(message_id))
            return

        message.extracted_knowledge = True

        ki_stmt = select(KnowledgeItem).where(KnowledgeItem.id == str(knowledge_item_id))
        ki_result = await self._session.execute(ki_stmt)
        knowledge_item = ki_result.scalar_one_or_none()
        if knowledge_item is not None:
            knowledge_item.source_message_id = str(message_id)
        else:
            logger.warning(
                "mark_knowledge_extracted_item_not_found",
                knowledge_item_id=str(knowledge_item_id),
            )

        await self._session.flush()

        logger.info(
            "knowledge_extracted_marked",
            message_id=str(message_id),
            knowledge_item_id=str(knowledge_item_id),
        )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_conversation_manager(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationManager:
    """FastAPI ``Depends`` factory for :class:`ConversationManager`.

    Usage::

        @router.post("/example")
        async def example(mgr: ConversationManager = Depends(get_conversation_manager)):
            conv = await mgr.create_or_get_conversation(session_id)
            ...
    """
    return ConversationManager(session)
