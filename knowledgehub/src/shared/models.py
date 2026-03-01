"""
SQLAlchemy 2.0 async models for KnowledgeHub.

All domain models live here so that Base.metadata has a single source of truth
for table definitions, migrations, and init_db.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.shared.database import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentType(str, enum.Enum):
    CONVERSATION_EXTRACT = "conversation_extract"
    DOCUMENT = "document"
    MANUAL = "manual"


class RuleType(str, enum.Enum):
    KEYWORD = "keyword"
    REGEX = "regex"
    SEMANTIC = "semantic"
    COMPOSITE = "composite"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Conversation
# ═══════════════════════════════════════════════════════════════════════════

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    session_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, comment="Open WebUI session link",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict, comment="User / context info",
    )

    # -- relationships -------------------------------------------------------
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_conversations_session_created", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id!r} session={self.session_id!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Message
# ═══════════════════════════════════════════════════════════════════════════

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role", native_enum=False, length=20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    detected_contexts: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=list, comment="List of triggered context ids",
    )
    extracted_knowledge: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether knowledge was extracted from this message",
    )

    # -- relationships -------------------------------------------------------
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    knowledge_items: Mapped[list["KnowledgeItem"]] = relationship(
        back_populates="source_message",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id", "created_at"),
        Index("ix_messages_role", "role"),
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id!r} role={self.role!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Context  (defined before KnowledgeItem / DetectionRule that reference it)
# ═══════════════════════════════════════════════════════════════════════════

class Context(Base):
    __tablename__ = "contexts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True,
        comment='e.g. "progetto_alpha", "procedura_onboarding"',
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("contexts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # -- self-referential relationship (hierarchy) ---------------------------
    parent: Mapped["Context | None"] = relationship(
        remote_side="Context.id",
        back_populates="children",
        lazy="selectin",
    )
    children: Mapped[list["Context"]] = relationship(
        back_populates="parent",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_contexts_parent", "parent_id"),
    )

    def __repr__(self) -> str:
        return f"<Context id={self.id!r} name={self.name!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 4. KnowledgeItem
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeItem(Base):
    __tablename__ = "knowledge_items"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    source_message_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="NULL when source is upload or manual entry",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[ContentType] = mapped_column(
        Enum(ContentType, name="content_type", native_enum=False, length=30),
        nullable=False,
    )
    contexts: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=list,
        comment='Context tags, e.g. ["progetto_x", "procedura_y"]',
    )
    embedding_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True,
        comment="Reference to the vector store entry",
    )
    verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Admin-approved flag",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    created_by: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="User or system identifier",
    )

    # -- relationships -------------------------------------------------------
    source_message: Mapped["Message | None"] = relationship(
        back_populates="knowledge_items",
    )

    __table_args__ = (
        Index("ix_knowledge_items_content_type", "content_type"),
        Index("ix_knowledge_items_verified", "verified"),
        Index("ix_knowledge_items_created_by", "created_by"),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeItem id={self.id!r} type={self.content_type!r} verified={self.verified}>"


# ═══════════════════════════════════════════════════════════════════════════
# 5. DetectionRule  (replaces the old version in knowledge/models.py)
# ═══════════════════════════════════════════════════════════════════════════

class DetectionRule(Base):
    __tablename__ = "detection_rules"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_type: Mapped[RuleType] = mapped_column(
        Enum(RuleType, name="rule_type", native_enum=False, length=20),
        nullable=False,
        default=RuleType.KEYWORD,
    )
    rule_config: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=dict,
        comment='Type-specific config: {"keywords": [...]} | {"pattern": "..."} | ...',
    )
    target_contexts: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=list,
        comment="Context ids/names to assign when this rule matches",
    )
    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    __table_args__ = (
        Index("ix_detection_rules_enabled_priority", "enabled", "priority"),
    )

    def __repr__(self) -> str:
        return f"<DetectionRule id={self.id!r} name={self.name!r} type={self.rule_type!r}>"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Document  (unchanged from original, kept for knowledge base uploads)
# ═══════════════════════════════════════════════════════════════════════════

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} title={self.title!r}>"
