# Troubleshooting

## Quick Diagnostics

Run the health check first:

```bash
python scripts/health_check.py --verbose
```

Check logs:

```bash
make logs                    # All services
make logs SERVICE=gateway    # Gateway only
```

---

## Gateway Not Responding

### Symptom: Connection refused on port 8000

**Check if the container is running:**

```bash
docker compose ps
```

If stopped, check logs:

```bash
docker compose logs gateway --tail=50
```

**Common causes:**

1. **Port conflict** — another service is using 8000

   ```bash
   lsof -i :8000
   # Fix: change GATEWAY_PORT in .env
   ```

2. **Database not initialized** — tables don't exist

   ```bash
   make shell SERVICE=gateway
   python scripts/init_db.py
   exit
   ```

3. **LLM backend unreachable** — Ollama not started

   ```bash
   curl http://localhost:11434/api/tags
   # If no response: docker compose restart ollama
   ```

### Symptom: 502 Bad Gateway from Open WebUI

Open WebUI can't reach the gateway. Check:

```bash
# Verify gateway is on the Docker network
docker network inspect knowledgehub_knowledgehub

# Test from inside the network
docker compose exec open-webui curl http://gateway:8000/health
```

**Fix:** Ensure `OLLAMA_BASE_URL=http://gateway:8000` in Open WebUI config.

---

## RAG Not Working

### Symptom: Bot doesn't use knowledge base

**Step 1: Verify detection works**

```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Come si configura il database?"}'
```

If `triggered_rules` is empty → no rules match. Check:

```bash
curl http://localhost:8001/api/v1/admin/rules \
  -H "X-API-Key: $API_KEY"
```

Ensure rules are `enabled: true` and keywords match your query.

**Step 2: Verify knowledge exists**

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database", "contexts": ["database"]}'
```

If empty → add knowledge:

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{"content": "Your knowledge text", "contexts": ["database"]}'
```

**Step 3: Verify embeddings**

If search returns no results even with knowledge added, the embedding may have failed:

```bash
# Check if items have embedding_id set
curl http://localhost:8001/api/v1/admin/knowledge \
  -H "X-API-Key: $API_KEY"
```

Items with `embedding_id: null` were not embedded. Re-add them.

### Symptom: RAG context injected but bot ignores it

The LLM model may not follow the system message well. Try:

1. Use a better model: `OLLAMA_MODEL=llama3.2:3b` instead of `phi3:mini`
2. Increase the RAG context priority in the prompt template

---

## Slow Performance

### Detection takes > 100ms

```bash
# Check how many rules are loaded
curl http://localhost:8001/api/v1/admin/analytics/overview \
  -H "X-API-Key: $API_KEY"
```

**Fixes:**

- Disable rules you don't need
- Avoid SemanticRule with many reference texts (each requires embedding)
- Increase `rule_timeout` if semantic rules are timing out

### LLM responses are slow

```bash
# Check Ollama GPU usage
docker compose exec ollama nvidia-smi  # If GPU available

# Check model size
docker compose exec ollama ollama list
```

**Fixes:**

- Use a smaller model: `phi3:mini` (3.8B) vs `llama3.2:3b`
- Reduce `max_tokens` in requests
- Enterprise: switch to vLLM for GPU-optimized inference

### Vector search is slow

```bash
# Check collection size
docker compose exec chroma curl http://localhost:8000/api/v1/collections
```

**Fixes:**

- Reduce `n_results` in search queries
- Enterprise: switch to Qdrant (optimized for large collections)
- Ensure embeddings dimension is consistent

### High memory usage

```bash
docker stats
```

**Fixes:**

- Reduce Docker resource limits in `docker-compose.yml`
- Use `embedding_batch_size=32` for mini profile
- Reduce `db_pool_size` and `db_max_overflow`

---

## Common Errors

### `sqlalchemy.exc.OperationalError: unable to open database file`

SQLite data directory doesn't exist or lacks permissions.

```bash
mkdir -p data/sqlite
chmod 755 data/sqlite
```

### `chromadb.errors.InvalidCollectionException`

ChromaDB metadata schema mismatch. Reset the collection:

```bash
# Warning: this deletes all vectors
rm -rf data/chroma
make shell SERVICE=gateway
python scripts/init_db.py
exit
```

### `httpx.ConnectError: Connection refused` (to Ollama)

Ollama service not running or not ready.

```bash
# Check Ollama status
docker compose logs ollama --tail=20

# Restart Ollama
docker compose restart ollama

# Wait for it and test
sleep 5
curl http://localhost:11434/api/tags
```

### `ImportError: No module named 'sentence_transformers'`

Missing dependency — install with full dev deps:

```bash
poetry install
```

### `RuntimeError: no running event loop`

Async function called from sync context. Ensure you use `asyncio.run()`:

```python
# Wrong
result = await some_async_function()

# Right
import asyncio
result = asyncio.run(some_async_function())
```

### `401 Unauthorized` on admin API

Missing or invalid API key.

```bash
# Check configured key
grep API_KEY .env

# Use it in requests
curl -H "X-API-Key: your-key" http://localhost:8001/api/v1/admin/rules
```

### `greenlet_spawn has not been called` (SQLAlchemy)

Accessing lazy-loaded relationships outside an async context. Use `lazy="selectin"` on relationships, or load data within the session:

```python
# In the query, eagerly load relationships
result = await session.execute(
    select(Conversation).options(selectinload(Conversation.messages))
)
```

### `ValueError: Invalid CompositeRule operator`

Only `AND`, `OR`, `NOT` operators are supported for composite rules.

---

## Docker Issues

### Container keeps restarting

```bash
docker compose logs gateway --tail=50
```

Common causes:
- Health check failing (curl not installed in image)
- Database connection error on startup
- Missing environment variables

### Build fails with "poetry.lock not found"

```bash
poetry lock
docker compose build --no-cache
```

### Out of disk space

```bash
# Clean Docker system
docker system prune -a --volumes

# Clean old backups
python scripts/backup.py --retention-days 7
```

---

## Getting Help

1. Check the [Architecture docs](ARCHITECTURE.md) to understand the system
2. Review the [API Reference](API_REFERENCE.md) for endpoint details
3. Run `python scripts/health_check.py --json` and share the output
4. Check gateway logs: `make logs SERVICE=gateway`
5. File an issue with:
   - KnowledgeHub version
   - Profile (mini/full)
   - Error message and full traceback
   - Steps to reproduce
