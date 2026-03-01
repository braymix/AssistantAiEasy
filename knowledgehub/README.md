# KnowledgeHub

AI-powered knowledge base with context detection and LLM integration.

## Profiles

| Feature        | `mini` (default)     | `full` (enterprise)       |
|----------------|----------------------|---------------------------|
| Database       | SQLite               | PostgreSQL                |
| Vector Store   | ChromaDB             | Qdrant                    |
| LLM Backend    | Ollama               | vLLM (GPU)                |
| Embeddings     | Sentence-Transformers | Sentence-Transformers    |

## Quick Start

```bash
cp .env.example .env
make install
make init
make dev
```

## Docker

```bash
# Mini profile (default)
make up PROFILE=mini

# Full enterprise profile
make up PROFILE=full

# Development with hot-reload
make up-dev
```

## Project Structure

- `src/gateway/` - Main FastAPI service (REST API)
- `src/knowledge/` - Knowledge base management and vector store
- `src/detection/` - Context detection engine with rules
- `src/llm/` - LLM provider abstraction (Ollama / vLLM)
- `src/admin/` - Admin dashboard
- `src/shared/` - Shared utilities, database, exceptions

## Development

```bash
make test       # Run tests with coverage
make lint       # Linter + type checks
make format     # Auto-format code
```
