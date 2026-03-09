# KnowledgeHub

**AI-powered corporate knowledge base that intercepts LLM conversations, detects context, and enriches responses with Retrieval-Augmented Generation (RAG).**

## What is KnowledgeHub?

KnowledgeHub sits between your chat frontend (Open WebUI) and your LLM backend (Ollama/vLLM) as an intelligent proxy. Every conversation passes through its Gateway, which:

1. **Detects context** — keyword, regex, semantic, and composite rules identify what topics the user is asking about
2. **Retrieves knowledge** — relevant documents are fetched from the vector store via semantic search
3. **Enriches the prompt** — retrieved knowledge is injected as a system message before the LLM generates a response
4. **Learns continuously** — key facts are extracted from conversations and stored for future use

```
┌──────────────┐     ┌──────────────────────────────────────────┐     ┌─────────┐
│              │     │            KnowledgeHub                  │     │         │
│  Open WebUI  │────>│  Gateway ──> Detection ──> RAG ──> LLM  │────>│ Ollama  │
│  (frontend)  │<────│    │            │            │           │<────│ / vLLM  │
│              │     │    v            v            v           │     │         │
└──────────────┘     │  ConvMgr    VectorStore   Knowledge DB  │     └─────────┘
                     │                                         │
                     │  Admin Dashboard (port 8001)            │
                     └──────────────────────────────────────────┘
```

## Use Cases

- **Corporate knowledge assistant** — employees ask questions, KnowledgeHub enriches answers with internal procedures, project docs, and HR policies
- **Onboarding chatbot** — new hires get contextual answers about company processes, drawing from a curated knowledge base
- **Technical support** — detects error codes and database issues, injects relevant troubleshooting guides
- **Document Q&A** — upload documents, ask questions, get answers grounded in your actual content

## Two Deployment Profiles

| Feature | Mini | Enterprise |
|---------|------|-----------|
| Database | SQLite | PostgreSQL |
| Vector store | ChromaDB | Qdrant |
| LLM backend | Ollama (CPU) | vLLM (GPU) |
| Embeddings | sentence-transformers (local) | Ollama API |
| Concurrency | 4 requests | 32 requests |
| Target | Single machine / Mini PC | Multi-node cluster |

## Quick Start

```bash
# 1. Clone and enter the project
cd knowledgehub

# 2. Copy environment config
cp .env.example .env

# 3. Start the mini stack
make up

# 4. Initialize the database and seed default rules
make shell SERVICE=gateway
python scripts/init_db.py --seed
exit

# 5. Open the interfaces
#    Chat:  http://localhost:3000  (Open WebUI)
#    Admin: http://localhost:8001  (Dashboard)
#    API:   http://localhost:8000  (Gateway)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](INSTALLATION.md) | Step-by-step setup for Mini and Enterprise |
| [Configuration](CONFIGURATION.md) | All environment variables and tuning options |
| [Architecture](ARCHITECTURE.md) | Component diagrams, data flows, design decisions |
| [Admin Guide](ADMIN_GUIDE.md) | Managing rules, contexts, knowledge, analytics |
| [User Guide](USER_GUIDE.md) | How to use the chatbot effectively |
| [API Reference](API_REFERENCE.md) | Complete endpoint docs with curl examples |
| [Development](DEVELOPMENT.md) | Dev setup, testing, code style, contributing |
| [Migration](MIGRATION.md) | Moving from Mini to Enterprise profile |
| [Troubleshooting](TROUBLESHOOTING.md) | Common problems and solutions |
| [Open WebUI Setup](OPEN_WEBUI_SETUP.md) | Pipeline and proxy mode configuration |
| [Proxy Mode](PROXY_MODE.md) | Using KnowledgeHub as an Ollama proxy |

## Tech Stack

- **Python 3.11+** with FastAPI, SQLAlchemy 2.0 (async), Pydantic v2
- **Ollama** / **vLLM** for LLM inference
- **ChromaDB** / **Qdrant** for vector search
- **Open WebUI** as the chat frontend
- **Docker Compose** for deployment
- **structlog** for structured JSON logging

## License

See [LICENSE](../LICENSE) for details.
