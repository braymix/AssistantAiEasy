# Migration Guide: Mini to Enterprise

This guide walks through migrating from the Mini profile (SQLite + ChromaDB + Ollama) to the Enterprise profile (PostgreSQL + Qdrant + vLLM).

---

## Pre-Migration Checklist

- [ ] Enterprise infrastructure is provisioned (PostgreSQL, Qdrant, GPU server)
- [ ] Current system is backed up (`python scripts/backup.py`)
- [ ] Docker images are built and tested on the target environment
- [ ] NVIDIA drivers and `nvidia-container-toolkit` are installed (for vLLM)
- [ ] Network connectivity between services is verified
- [ ] DNS/load balancer is configured (if applicable)
- [ ] Maintenance window is communicated to users

---

## Step-by-Step Migration

### Phase 1: Backup (15 min)

```bash
# Full backup of current mini deployment
python scripts/backup.py --output /tmp/migration_backup

# Verify backup files exist
ls -la /tmp/migration_backup/
```

### Phase 2: Export Data (10 min)

```bash
# Export knowledge base
curl http://localhost:8001/api/v1/admin/knowledge/export \
  -H "X-API-Key: $API_KEY" > /tmp/knowledge_export.json

# Export rules
curl http://localhost:8001/api/v1/admin/rules \
  -H "X-API-Key: $API_KEY" > /tmp/rules_export.json

# Export contexts
curl http://localhost:8001/api/v1/admin/contexts \
  -H "X-API-Key: $API_KEY" > /tmp/contexts_export.json
```

### Phase 3: Configure Enterprise Environment (15 min)

```bash
# Copy and edit .env for enterprise
cp .env.example .env

# Edit .env:
KNOWLEDGEHUB_PROFILE=full

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://knowledgehub:secure-password@postgres:5432/knowledgehub
POSTGRES_DB=knowledgehub
POSTGRES_USER=knowledgehub
POSTGRES_PASSWORD=secure-password

# Qdrant
VECTORSTORE_BACKEND=qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# vLLM
LLM_BACKEND=vllm
VLLM_BASE_URL=http://vllm:8000
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_GPU_MEMORY=0.9

# Embeddings (switch to Ollama for higher quality)
KNOWLEDGEHUB_EMBEDDING_BACKEND=ollama
KNOWLEDGEHUB_OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# Security
SECRET_KEY=$(openssl rand -hex 32)
ADMIN_API_KEY=$(openssl rand -hex 16)
```

### Phase 4: Start Enterprise Infrastructure (10 min)

```bash
# Start enterprise stack
make up-enterprise

# Wait for all services to be healthy
python scripts/health_check.py --verbose
```

### Phase 5: Initialize Database (5 min)

```bash
make shell SERVICE=gateway

# Create tables
python scripts/init_db.py

# Seed default rules and contexts
python scripts/seed_rules.py

exit
```

### Phase 6: Migrate Data (20 min)

#### Option A: SQLite to PostgreSQL (direct SQL)

```bash
# Export SQLite data
sqlite3 data/sqlite/knowledgehub.db .dump > /tmp/sqlite_dump.sql

# Clean up SQLite-specific syntax for PostgreSQL
sed -i 's/AUTOINCREMENT//' /tmp/sqlite_dump.sql
sed -i 's/INTEGER PRIMARY KEY/SERIAL PRIMARY KEY/' /tmp/sqlite_dump.sql

# Import into PostgreSQL
docker compose exec -T postgres psql -U knowledgehub -d knowledgehub < /tmp/sqlite_dump.sql
```

#### Option B: API-based migration (recommended)

```bash
# Re-import knowledge via admin API
curl -X POST http://localhost:8001/api/v1/admin/knowledge/import \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/knowledge_export.json
```

This re-embeds all knowledge with the new embedding model (768-dim instead of 384-dim).

### Phase 7: Re-embed Knowledge (10 min)

Since the embedding dimension changes (384 → 768), all vectors must be regenerated:

```bash
make shell SERVICE=gateway

python -c "
import asyncio
from src.config import get_settings
from src.shared.database import init_db, dispose_engine, AsyncSessionLocal
from src.knowledge.service import KnowledgeService
from src.knowledge.vectorstore import get_vector_store
from src.knowledge.embeddings import get_embedding_provider
from sqlalchemy import select
from src.shared.models import KnowledgeItem

async def reembed():
    await init_db()
    async with AsyncSessionLocal() as session:
        svc = KnowledgeService(
            session=session,
            vectorstore=get_vector_store(),
            embedder=get_embedding_provider(),
        )
        items = (await session.execute(select(KnowledgeItem))).scalars().all()
        for item in items:
            embedding = await svc._embedder.embed([item.content])
            await svc._vectorstore.add(
                texts=[item.content],
                metadatas=[{'contexts': str(item.contexts)}],
                ids=[item.embedding_id or item.id],
                embeddings=embedding,
            )
        await session.commit()
        print(f'Re-embedded {len(items)} items')
    await dispose_engine()

asyncio.run(reembed())
"

exit
```

### Phase 8: Verify (10 min)

```bash
# Health check
python scripts/health_check.py --verbose

# Test detection
curl -X POST http://localhost:8000/api/v1/detection/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Come si configura il database?"}'

# Test chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Verify analytics
curl http://localhost:8001/api/v1/admin/analytics/overview \
  -H "X-API-Key: $API_KEY"
```

### Phase 9: Switch Traffic (5 min)

Update Open WebUI to point to the new gateway:

```bash
# In docker-compose or Open WebUI settings:
OLLAMA_BASE_URL=http://new-gateway:8000
```

---

## Rollback Plan

If something goes wrong, revert to mini:

### Quick Rollback (< 5 min)

```bash
# Stop enterprise stack
make down-enterprise

# Restore .env to mini settings
cp .env.backup .env

# Restore SQLite from backup
cp /tmp/migration_backup/knowledgehub_backup_db_*.sqlite.gz data/sqlite/
cd data/sqlite && gunzip knowledgehub_backup_db_*.sqlite.gz
mv knowledgehub_backup_db_*.sqlite knowledgehub.db
cd ../..

# Start mini stack
make up

# Verify
python scripts/health_check.py
```

### Data Rollback

If data was corrupted:

```bash
# Restore ChromaDB
tar xzf /tmp/migration_backup/knowledgehub_backup_chroma_*.tar.gz -C data/

# Restore SQLite
gunzip -k /tmp/migration_backup/knowledgehub_backup_db_*.sqlite.gz
cp /tmp/migration_backup/knowledgehub_backup_db_*.sqlite data/sqlite/knowledgehub.db
```

---

## Post-Migration Validation

| Check | Command | Expected |
|-------|---------|----------|
| All services healthy | `python scripts/health_check.py` | Exit code 0 |
| Database tables exist | `python scripts/migrate.py status` | 6 tables |
| Rules loaded | `curl .../admin/analytics/overview` | `total_rules > 0` |
| Knowledge searchable | `curl .../knowledge/search` | Results returned |
| Chat works | `curl .../v1/chat/completions` | LLM response |
| Streaming works | `curl .../v1/chat/completions` (stream=true) | SSE chunks |
| Admin dashboard | Open `http://localhost:8001` | Page loads |

---

## Performance Comparison

After migration, you should see improvements:

| Metric | Mini | Enterprise |
|--------|------|-----------|
| Chat latency (first token) | 2-5s | 0.5-1s |
| Detection latency | < 50ms | < 20ms |
| Concurrent users | 2-4 | 20-50 |
| Knowledge search | < 200ms | < 50ms |
| Embedding speed | 10 docs/s | 100 docs/s |
