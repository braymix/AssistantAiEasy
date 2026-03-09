# Configuration Reference

All configuration is managed via environment variables. Copy `.env.example` to `.env` and customize.

Settings are loaded by `pydantic-settings` with the `KNOWLEDGEHUB_` prefix for app-level vars.

---

## Profile Selection

```bash
KNOWLEDGEHUB_PROFILE=mini   # mini | full
```

| Setting | Mini | Full |
|---------|------|------|
| Database | SQLite (aiosqlite) | PostgreSQL (asyncpg) |
| Vector store | ChromaDB | Qdrant |
| LLM backend | Ollama | vLLM |
| Embeddings | sentence-transformers (local) | Ollama API |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | nomic-embed-text (768-dim) |
| Max concurrency | 4 | 32 |
| Chunk size | 512 tokens | 1024 tokens |
| Chunk overlap | 50 tokens | 100 tokens |

---

## Database

```bash
# Mini (SQLite)
DATABASE_URL=sqlite+aiosqlite:///./data/sqlite/knowledgehub.db

# Full (PostgreSQL)
DATABASE_URL=postgresql+asyncpg://knowledgehub:changeme@postgres:5432/knowledgehub
POSTGRES_DB=knowledgehub
POSTGRES_USER=knowledgehub
POSTGRES_PASSWORD=changeme
```

Connection pool settings (PostgreSQL):

```bash
KNOWLEDGEHUB_DB_POOL_SIZE=5       # Base pool size
KNOWLEDGEHUB_DB_MAX_OVERFLOW=10   # Extra connections under load
KNOWLEDGEHUB_DB_ECHO=false        # SQL query logging
```

---

## Vector Store

```bash
# ChromaDB (mini)
VECTORSTORE_BACKEND=chroma
CHROMA_HOST=chroma
CHROMA_PORT=8100
CHROMA_PERSIST_DIR=./data/chroma
KNOWLEDGEHUB_CHROMA_COLLECTION=knowledgehub

# Qdrant (full)
VECTORSTORE_BACKEND=qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
KNOWLEDGEHUB_QDRANT_COLLECTION=knowledgehub
```

---

## LLM Backend

```bash
# Ollama (mini)
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=phi3:mini
KNOWLEDGEHUB_OLLAMA_TIMEOUT=120    # Request timeout in seconds

# vLLM (full)
LLM_BACKEND=vllm
VLLM_BASE_URL=http://vllm:8000
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
KNOWLEDGEHUB_VLLM_MAX_TOKENS=4096
```

---

## Embeddings

```bash
# Local sentence-transformers (mini)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
KNOWLEDGEHUB_EMBEDDING_BATCH_SIZE=64

# Ollama embeddings (full)
KNOWLEDGEHUB_EMBEDDING_BACKEND=ollama
KNOWLEDGEHUB_OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
KNOWLEDGEHUB_EMBEDDING_BATCH_SIZE=128
```

---

## Service Ports

```bash
GATEWAY_PORT=8000       # Gateway API
ADMIN_PORT=8001         # Admin dashboard
OLLAMA_PORT=11434       # Ollama API
CHROMA_PORT=8100        # ChromaDB
OPENWEBUI_PORT=3000     # Open WebUI frontend
QDRANT_PORT=6333        # Qdrant (enterprise)
POSTGRES_PORT=5432      # PostgreSQL (enterprise)
VLLM_PORT=8080          # vLLM (enterprise)
REDIS_PORT=6379         # Redis (enterprise)
```

---

## Open WebUI Integration

```bash
OPENWEBUI_URL=http://open-webui:8080
OPENWEBUI_API_KEY=                    # Bearer token for Open WebUI API
OPENWEBUI_AUTH=true                   # Enable auth in Open WebUI
OPENWEBUI_SECRET=changeme            # Open WebUI secret key
```

---

## Logging

```bash
LOG_LEVEL=INFO           # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=json          # json | console
```

- `json`: structured JSON logs (recommended for production)
- `console`: human-readable colored output (recommended for dev)

---

## Security

```bash
SECRET_KEY=change-this-to-a-random-secret-key
API_KEY_HEADER=X-API-Key
ADMIN_API_KEY=change-me-in-production
```

Generate a secure key:

```bash
openssl rand -hex 32
```

### Security Hardening Checklist

1. **Change all default passwords** — `SECRET_KEY`, `POSTGRES_PASSWORD`, `ADMIN_API_KEY`, `OPENWEBUI_SECRET`
2. **Restrict network access** — only expose ports 3000 (frontend) and 8001 (admin) externally
3. **Use TLS** — put a reverse proxy (nginx/Caddy) in front with HTTPS
4. **Limit API keys** — configure `API_KEYS=key1,key2` to restrict gateway access
5. **Enable auth in Open WebUI** — set `OPENWEBUI_AUTH=true`
6. **Run as non-root** — the Docker images use a `khub` user by default
7. **Rotate credentials** — change API keys periodically
8. **Monitor logs** — use `LOG_FORMAT=json` with a log aggregator (ELK, Loki)

---

## Performance Tuning

### Mini Profile

For a machine with 4 cores / 8 GB RAM:

```bash
KNOWLEDGEHUB_MAX_CONCURRENT_REQUESTS=4
OLLAMA_MODEL=phi3:mini              # Smallest model
KNOWLEDGEHUB_CHUNK_SIZE=512
KNOWLEDGEHUB_EMBEDDING_BATCH_SIZE=32
```

### Enterprise Profile

For a server with 16+ cores / 64 GB RAM / GPU:

```bash
KNOWLEDGEHUB_MAX_CONCURRENT_REQUESTS=32
KNOWLEDGEHUB_DB_POOL_SIZE=20
KNOWLEDGEHUB_DB_MAX_OVERFLOW=30
KNOWLEDGEHUB_CHUNK_SIZE=1024
KNOWLEDGEHUB_EMBEDDING_BATCH_SIZE=256
VLLM_GPU_MEMORY=0.9
VLLM_MAX_MODEL_LEN=8192
```

### Docker Resource Limits

In `docker-compose.yml`, adjust the `deploy.resources` section:

```yaml
gateway:
  deploy:
    resources:
      limits:
        memory: 1G    # Increase for high traffic
        cpus: "2.0"
```

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWLEDGEHUB_PROFILE` | `mini` | Deployment profile: `mini` or `full` |
| `DATABASE_URL` | SQLite path | Database connection string |
| `VECTORSTORE_BACKEND` | `chroma` | Vector store: `chroma` or `qdrant` |
| `LLM_BACKEND` | `ollama` | LLM provider: `ollama` or `vllm` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `phi3:mini` | Default Ollama model |
| `VLLM_BASE_URL` | `http://vllm:8000` | vLLM API URL |
| `VLLM_MODEL` | `Llama-3.1-8B-Instruct` | Default vLLM model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector dimension |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `LOG_FORMAT` | `json` | Log output format |
| `SECRET_KEY` | (required) | App secret for signing |
| `ADMIN_API_KEY` | `change-me-in-production` | Admin API authentication key |
| `OPENWEBUI_URL` | `http://open-webui:8080` | Open WebUI base URL |
| `OPENWEBUI_API_KEY` | (empty) | Open WebUI API token |
