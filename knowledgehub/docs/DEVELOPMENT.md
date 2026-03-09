# Development Guide

## Dev Environment Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) 1.8+
- Docker (for external services)
- Git

### 1. Install dependencies

```bash
cd knowledgehub
poetry install    # Installs all deps including dev group
```

### 2. Start external services

```bash
# Option A: Use Docker for Ollama + ChromaDB
docker compose up ollama chroma -d

# Option B: Run Ollama natively
ollama serve &
ollama pull phi3:mini
```

### 3. Run the gateway

```bash
make dev          # Starts gateway on :8000 with hot reload
```

### 4. Run the admin dashboard

```bash
make dev-admin    # Starts admin on :8001 with hot reload
```

### 5. Run with Docker (dev mode)

```bash
make up-dev       # Hot reload + debug ports (5678, 5679)
```

---

## Project Structure

```
knowledgehub/
├── src/
│   ├── admin/           # Admin dashboard + REST API
│   │   ├── routes/      # FastAPI route handlers
│   │   ├── schemas/     # Pydantic request/response models
│   │   └── dependencies.py  # API key auth
│   ├── config/          # Settings, logging
│   ├── detection/       # Context detection engine
│   │   ├── engine.py    # DetectionEngine orchestrator
│   │   ├── rules.py     # Rule ABC + implementations
│   │   └── triggers.py  # Actions (enrich, tag, log)
│   ├── gateway/         # Gateway API (OpenAI proxy)
│   │   ├── routes/      # chat, health, detection, knowledge, query
│   │   ├── services/    # ConversationManager, SessionSync
│   │   └── schemas/     # OpenAI-compatible schemas
│   ├── knowledge/       # Knowledge base
│   │   ├── service.py   # KnowledgeService
│   │   ├── vectorstore.py  # VectorStore ABC + Chroma/Qdrant
│   │   └── embeddings.py   # EmbeddingProvider ABC
│   ├── llm/             # LLM abstraction
│   │   ├── base.py      # LLMProvider ABC
│   │   ├── ollama.py    # OllamaProvider
│   │   ├── vllm.py      # VLLMProvider
│   │   ├── factory.py   # Singleton factory
│   │   ├── rag.py       # RAGOrchestrator
│   │   └── prompts.py   # Prompt templates
│   └── shared/          # Shared code
│       ├── database.py  # AsyncEngine + session factory
│       ├── models.py    # SQLAlchemy models
│       └── exceptions.py
├── tests/               # Test suite
├── scripts/             # CLI scripts (init, seed, migrate, backup)
├── docs/                # Documentation
├── pipelines/           # Open WebUI pipeline
├── Dockerfile           # Multi-stage build
├── docker-compose.yml   # Mini stack
├── docker-compose.enterprise.yml
├── docker-compose.dev.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Code Style

### Linter and Formatter

We use **Ruff** for both linting and formatting:

```bash
make lint         # Check for issues
make format       # Auto-fix formatting
```

Ruff config in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]
```

### Type Checking

```bash
poetry run mypy src/
```

### Conventions

- **Async everywhere** — all DB and HTTP operations use `async/await`
- **Pydantic v2** for all schemas and config
- **SQLAlchemy 2.0** mapped columns (no legacy `Column()` syntax)
- **structlog** for logging — use `get_logger(__name__)`, log events as snake_case
- **No bare exceptions** — always catch specific exception types
- **Type hints** on all function signatures

---

## Testing

### Run tests

```bash
make test         # Full suite with coverage
make test-fast    # Quick run, stop on first failure
```

### Test structure

```
tests/
├── conftest.py              # Shared fixtures (db, client, mocks)
├── test_detection/
│   ├── test_rules.py        # Rule matching logic
│   ├── test_engine.py       # DetectionEngine orchestration
│   ├── test_triggers.py     # Action system
│   └── test_action_registry.py
├── test_gateway/
│   ├── test_chat.py         # Chat endpoint + schemas
│   ├── test_health.py       # Health endpoints
│   ├── test_openwebui_client.py
│   ├── test_session_sync.py
│   └── test_conversation_manager.py
├── test_knowledge/
│   ├── test_service.py      # KnowledgeService
│   └── test_vectorstore.py  # ChromaVectorStore
├── test_llm/
│   ├── test_ollama.py
│   ├── test_vllm.py
│   ├── test_factory.py
│   ├── test_rag.py
│   ├── test_prompts.py
│   └── test_models.py
├── test_admin/
│   ├── conftest.py          # Admin client with API key
│   ├── test_rules.py
│   ├── test_contexts.py
│   └── test_knowledge.py
├── test_integration/
│   └── test_full_flow.py    # End-to-end flows
└── test_performance/
    └── test_latency.py      # Latency benchmarks
```

### Key fixtures

| Fixture | Description |
|---------|-------------|
| `db_engine` | In-memory SQLite with all tables |
| `db_session` | AsyncSession for test isolation |
| `client` | HTTPx AsyncClient for gateway |
| `mock_llm_provider` | AsyncMock LLM (chat, stream, health) |
| `mock_vector_store` | In-memory FakeVectorStore |
| `mock_embedder` | Deterministic FakeEmbedder |
| `sample_rules` | 5 seeded DetectionRule rows |
| `sample_knowledge` | 3 seeded KnowledgeItem rows |
| `sample_contexts` | 5 seeded Context rows |

### Writing tests

```python
@pytest.mark.asyncio
async def test_my_feature(db_session, mock_llm_provider):
    """Test description."""
    # Arrange
    service = KnowledgeService(session=db_session, vectorstore=FakeVectorStore())

    # Act
    result = await service.add_knowledge(content="Test", contexts=["ctx"])

    # Assert
    assert result.id is not None
```

### Coverage target

The CI enforces `--cov-fail-under=80`. Check coverage locally:

```bash
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Debugging

### VS Code launch config

Add to `.vscode/launch.json`:

```json
{
  "configurations": [
    {
      "name": "Gateway",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
      "env": {"KNOWLEDGEHUB_PROFILE": "mini"}
    }
  ]
}
```

### Docker debug

Dev compose exposes debugpy ports:

- Gateway: `localhost:5678`
- Admin: `localhost:5679`

Attach with VS Code "Remote Attach" configuration.

---

## Database Migrations

```bash
python scripts/migrate.py upgrade   # Apply schema changes
python scripts/migrate.py status    # Show current tables
python scripts/migrate.py reset     # Drop all + recreate (destructive)
```

When adding a new model:

1. Add the class to `src/shared/models.py`
2. Run `python scripts/migrate.py upgrade` — creates missing tables
3. For column changes in existing tables, manually write ALTER statements or use Alembic

---

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes following code style guidelines
3. Add tests for new functionality
4. Run `make lint && make test`
5. Commit with descriptive message
6. Open a pull request

### Commit messages

Use conventional commits:

```
feat: add semantic rule caching
fix: handle empty vectorstore response
docs: update API reference
test: add integration tests for RAG flow
refactor: simplify detection engine
```
