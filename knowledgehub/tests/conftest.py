"""Shared test fixtures for the KnowledgeHub test suite.

Provides:
  - db_engine / db_session:    In-memory SQLite async engine + session
  - client:                    HTTPx async client for gateway app
  - mock_llm_provider:         AsyncMock LLM with chat/chat_stream/health_check
  - mock_vector_store:         In-memory VectorStore implementation
  - mock_embedder:             Deterministic fake embedding provider
  - sample_rules:              Pre-seeded DetectionRule rows in DB
  - sample_knowledge:          Pre-seeded KnowledgeItem rows in DB
  - sample_contexts:           Pre-seeded Context rows in DB
  - sample_conversation:       Pre-seeded Conversation with messages
"""

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Force mini profile with in-memory SQLite for tests
os.environ.setdefault("KNOWLEDGEHUB_PROFILE", "mini")
os.environ.setdefault("KNOWLEDGEHUB_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from src.shared.database import Base  # noqa: E402
import src.shared.models  # noqa: E402, F401 – register all models on Base.metadata
from src.shared.models import (  # noqa: E402
    ContentType,
    Context,
    Conversation,
    DetectionRule,
    KnowledgeItem,
    Message,
    MessageRole,
    RuleType,
)
from src.knowledge.vectorstore import DocumentRecord, SearchResult, VectorStore  # noqa: E402
from src.knowledge.embeddings import EmbeddingProvider  # noqa: E402

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# ═══════════════════════════════════════════════════════════════════════════════
# Database fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def db_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP client fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    from src.gateway.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ═══════════════════════════════════════════════════════════════════════════════
# Mock LLM provider
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_llm_provider():
    """AsyncMock LLM provider with pre-configured return values."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="This is a test response from the LLM.")

    async def _fake_stream(*args, **kwargs):
        for token in ["This ", "is ", "a ", "test ", "response."]:
            yield token

    llm.chat_stream = MagicMock(side_effect=lambda *a, **kw: _fake_stream())

    llm.generate = AsyncMock(return_value="Generated text.")

    async def _fake_gen_stream(*args, **kwargs):
        for token in ["Generated ", "text."]:
            yield token

    llm.generate_stream = MagicMock(side_effect=lambda *a, **kw: _fake_gen_stream())

    llm.health_check = AsyncMock(return_value=MagicMock(healthy=True, latency_ms=5.0))

    return llm


# ═══════════════════════════════════════════════════════════════════════════════
# Mock vector store
# ═══════════════════════════════════════════════════════════════════════════════


class FakeVectorStore(VectorStore):
    """In-memory vector store for tests — no external dependencies."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def add(self, texts, metadatas, ids, embeddings=None) -> list[str]:
        for uid, text, meta in zip(ids, texts, metadatas):
            self._store[uid] = {"text": text, "metadata": meta, "embedding": embeddings}
        return ids

    async def search(self, query_embedding, n_results=5, filter=None) -> list[SearchResult]:
        results = []
        for uid, data in list(self._store.items())[:n_results]:
            if filter:
                skip = False
                for k, v in filter.items():
                    if data["metadata"].get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            results.append(SearchResult(
                id=uid, content=data["text"], score=0.85, metadata=data["metadata"],
            ))
        return results

    async def delete(self, ids) -> bool:
        for uid in ids:
            self._store.pop(uid, None)
        return True

    async def get(self, ids) -> list[DocumentRecord]:
        return [
            DocumentRecord(id=uid, content=self._store[uid]["text"], metadata=self._store[uid]["metadata"])
            for uid in ids if uid in self._store
        ]

    async def update(self, id, text=None, metadata=None, embedding=None) -> bool:
        if id not in self._store:
            return False
        if text is not None:
            self._store[id]["text"] = text
        if metadata is not None:
            self._store[id]["metadata"].update(metadata)
        return True


@pytest.fixture
def mock_vector_store():
    return FakeVectorStore()


# ═══════════════════════════════════════════════════════════════════════════════
# Mock embedder
# ═══════════════════════════════════════════════════════════════════════════════


class FakeEmbedder(EmbeddingProvider):
    """Returns deterministic fake embeddings."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[0.1 * (i + 1)] * self._dim for i, _ in enumerate(texts)]

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def mock_embedder():
    return FakeEmbedder()


# ═══════════════════════════════════════════════════════════════════════════════
# Sample data fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def sample_contexts(db_session) -> list[Context]:
    """Seed contexts for tests."""
    contexts = [
        Context(name="projects", description="Progetti aziendali"),
        Context(name="procedures", description="Procedure operative"),
        Context(name="onboarding", description="Formazione nuovi dipendenti"),
        Context(name="database", description="Database topics"),
        Context(name="errors", description="Error codes and traces"),
    ]
    for ctx in contexts:
        db_session.add(ctx)
    await db_session.flush()
    return contexts


@pytest.fixture
async def sample_rules(db_session, sample_contexts) -> list[DetectionRule]:
    """Seed detection rules for tests."""
    rules = [
        DetectionRule(
            name="project_mention",
            description="Rileva menzioni di progetti",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": ["progetto", "project"], "case_sensitive": False},
            target_contexts=["projects"],
            priority=10,
            enabled=True,
        ),
        DetectionRule(
            name="procedure_request",
            description="Rileva richieste di procedure",
            rule_type=RuleType.REGEX,
            rule_config={"patterns": [r"come (si fa|faccio|posso)", r"procedura per"]},
            target_contexts=["procedures"],
            priority=10,
            enabled=True,
        ),
        DetectionRule(
            name="error_codes",
            description="Rileva codici errore",
            rule_type=RuleType.REGEX,
            rule_config={"pattern": r"(ERR[-_]\d+|HTTP\s+[45]\d{2})"},
            target_contexts=["errors"],
            priority=9,
            enabled=True,
        ),
        DetectionRule(
            name="database_issues",
            description="Database questions",
            rule_type=RuleType.COMPOSITE,
            rule_config={
                "keywords": ["database", "sql", "query"],
                "pattern": r"(?i)(db|database)\s+(error|issue|problem)",
                "operator": "OR",
            },
            target_contexts=["database"],
            priority=10,
            enabled=True,
        ),
        DetectionRule(
            name="disabled_rule",
            rule_type=RuleType.KEYWORD,
            rule_config={"keywords": ["secret"]},
            target_contexts=["security"],
            priority=100,
            enabled=False,
        ),
    ]
    for rule in rules:
        db_session.add(rule)
    await db_session.flush()
    return rules


@pytest.fixture
async def sample_knowledge(db_session, sample_contexts) -> list[KnowledgeItem]:
    """Seed knowledge items for tests."""
    items = [
        KnowledgeItem(
            content="Il progetto Alpha usa FastAPI come backend principale.",
            content_type=ContentType.MANUAL,
            contexts=["projects"],
            verified=True,
            created_by="admin",
        ),
        KnowledgeItem(
            content="La procedura di onboarding richiede 3 giorni lavorativi.",
            content_type=ContentType.MANUAL,
            contexts=["onboarding", "procedures"],
            verified=True,
            created_by="admin",
        ),
        KnowledgeItem(
            content="Per connettersi al database usare connection pooling con max 20 connessioni.",
            content_type=ContentType.CONVERSATION_EXTRACT,
            contexts=["database"],
            verified=False,
            created_by="system",
        ),
    ]
    for item in items:
        db_session.add(item)
    await db_session.flush()
    return items


@pytest.fixture
async def sample_conversation(db_session) -> Conversation:
    """Seed a conversation with messages for tests."""
    conv = Conversation(session_id="test-session-001", metadata_json={"user": "tester"})
    db_session.add(conv)
    await db_session.flush()

    messages = [
        Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Come si configura il database?",
            detected_contexts=["database"],
        ),
        Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="Per configurare il database, usa il file .env con DATABASE_URL.",
        ),
        Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Qual è la procedura per il deploy?",
            detected_contexts=["procedures"],
        ),
    ]
    for msg in messages:
        db_session.add(msg)
    await db_session.flush()

    return conv
