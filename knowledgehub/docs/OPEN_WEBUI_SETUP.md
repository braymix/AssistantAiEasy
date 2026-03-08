# Open WebUI + KnowledgeHub Setup Guide

This guide covers how to integrate [Open WebUI](https://github.com/open-webui/open-webui) with the KnowledgeHub Gateway using the **Pipeline** method.

> For the simpler proxy-based alternative (no pipeline needed), see [PROXY_MODE.md](./PROXY_MODE.md).

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌─────────┐
│  Open WebUI  │────▶│  KnowledgeHub        │────▶│  Ollama  │
│  (Frontend)  │◀────│  Gateway (:8000)     │◀────│  / vLLM  │
│  :3000       │     │  Detection + RAG     │     │  :11434  │
└──────────────┘     └──────────────────────┘     └─────────┘
       │                      │
       │ Pipeline              │
       │ (intercepts chat)     │ Vector Store
       ▼                      ▼
┌──────────────┐     ┌──────────────────────┐
│  Pipeline:   │     │  ChromaDB / Qdrant   │
│  knowledgehub│     │  :8100 / :6333       │
│  _pipeline   │     └──────────────────────┘
└──────────────┘
```

## Prerequisites

- Docker and Docker Compose installed
- KnowledgeHub stack running (gateway on port 8000)
- At least 4 GB RAM available for Open WebUI

## 1. Quick Start (Docker Compose)

The fastest way is to use the provided setup script:

```bash
cd knowledgehub/
bash scripts/setup_openwebui.sh
```

This will:
1. Add Open WebUI to `docker-compose.openwebui.yml`
2. Copy the pipeline into the container volume
3. Start everything

## 2. Manual Docker Setup

### 2.1 Start Open WebUI with Docker

```bash
docker run -d \
  --name open-webui \
  --network knowledgehub_knowledgehub \
  -p 3000:8080 \
  -e WEBUI_AUTH=false \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -v open-webui-data:/app/backend/data \
  -v ./pipelines:/app/backend/pipelines \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

> **Note**: The `--network` flag connects Open WebUI to the KnowledgeHub Docker network so it can reach the gateway and Ollama.

### 2.2 Install the Pipeline

**Option A — Volume mount (recommended for development):**

The `-v ./pipelines:/app/backend/pipelines` flag in the command above mounts the `pipelines/` directory directly. The pipeline is available immediately.

**Option B — Upload via Admin UI:**

1. Open http://localhost:3000
2. Go to **Admin Panel** → **Settings** → **Pipelines**
3. Click **Upload Pipeline**
4. Select `pipelines/knowledgehub_pipeline.py`

**Option C — Copy into running container:**

```bash
docker cp pipelines/knowledgehub_pipeline.py open-webui:/app/backend/pipelines/
docker restart open-webui
```

### 2.3 Configure the Pipeline

1. Open http://localhost:3000
2. Navigate to **Admin Panel** → **Settings** → **Pipelines**
3. Find **KnowledgeHub Pipeline** in the list
4. Click the gear icon to configure **Valves**:

| Valve | Default | Description |
|-------|---------|-------------|
| `gateway_url` | `http://localhost:8000` | KnowledgeHub Gateway URL. Use `http://gateway:8000` if running in Docker on the same network |
| `enable_rag` | `true` | Enable/disable RAG enrichment |
| `show_sources` | `true` | Append source references to responses |
| `min_confidence` | `0.7` | Minimum confidence for knowledge retrieval |
| `request_timeout` | `120` | HTTP timeout in seconds |
| `fallback_to_direct` | `true` | Fall back to default model if gateway is down |
| `api_key` | *(empty)* | API key for authenticated gateway access |

> **Important**: If both Open WebUI and KnowledgeHub run in Docker, set `gateway_url` to the Docker service name: `http://gateway:8000`

## 3. Environment Variables

You can configure the pipeline via environment variables instead of the UI:

```bash
# In your docker-compose or .env file
KNOWLEDGEHUB_GATEWAY_URL=http://gateway:8000
KNOWLEDGEHUB_API_KEY=your-secret-key
```

### Full Environment Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOWLEDGEHUB_GATEWAY_URL` | `http://localhost:8000` | Gateway base URL |
| `KNOWLEDGEHUB_API_KEY` | *(empty)* | API key for gateway auth |
| `KNOWLEDGEHUB_PROFILE` | `mini` | KnowledgeHub profile (`mini` or `full`) |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama URL (for fallback) |

## 4. Docker Compose (Full Stack)

To run everything together, use the provided override file:

```yaml
# docker-compose.openwebui.yml
version: "3.9"

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - WEBUI_AUTH=false
      - OLLAMA_BASE_URL=http://ollama:11434
      - ENABLE_OPENAI_API=true
      - OPENAI_API_BASE_URL=http://gateway:8000/v1
      - OPENAI_API_KEY=knowledgehub
    volumes:
      - open-webui-data:/app/backend/data
      - ./pipelines:/app/backend/pipelines
    depends_on:
      - gateway
      - ollama
    restart: unless-stopped
    networks:
      - knowledgehub

volumes:
  open-webui-data:

networks:
  knowledgehub:
    external: true
```

Start with:

```bash
# Start KnowledgeHub first
docker compose up -d

# Then start Open WebUI
docker compose -f docker-compose.openwebui.yml up -d
```

Or combine them:

```bash
docker compose -f docker-compose.yml -f docker-compose.openwebui.yml up -d
```

## 5. Verify the Integration

### 5.1 Check Gateway Health

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "profile": "mini",
  "llm_backend": "ollama",
  "vectorstore": "chroma",
  "version": "0.1.0"
}
```

### 5.2 Check Pipeline Status

1. Open http://localhost:3000
2. Go to **Admin Panel** → **Settings** → **Pipelines**
3. The KnowledgeHub Pipeline should show a green status indicator

### 5.3 Test a Chat Message

1. Start a new chat in Open WebUI
2. Type a message related to one of your configured knowledge contexts
3. The response should include RAG-enriched content from KnowledgeHub

## 6. Troubleshooting

### Pipeline not appearing in Open WebUI

- Verify the file is in the pipelines directory:
  ```bash
  docker exec open-webui ls /app/backend/pipelines/
  ```
- Check Open WebUI logs:
  ```bash
  docker logs open-webui 2>&1 | grep -i pipeline
  ```
- Restart Open WebUI after copying the pipeline file

### Gateway connection refused

- Verify the gateway is running:
  ```bash
  curl http://localhost:8000/health
  ```
- If using Docker, ensure both containers are on the same network:
  ```bash
  docker network inspect knowledgehub_knowledgehub
  ```
- Use the Docker service name (`http://gateway:8000`) instead of `localhost`

### Slow responses

- Check Ollama model is loaded:
  ```bash
  curl http://localhost:11434/api/tags
  ```
- Verify vector store connectivity:
  ```bash
  curl http://localhost:8100/api/v1/heartbeat  # ChromaDB
  ```
- Increase `request_timeout` in the pipeline Valves

### Streaming not working

- Ensure the pipeline returns a generator (streaming mode is the default)
- Check for proxy/reverse-proxy buffering (nginx, Cloudflare, etc.)
- Verify `X-Accel-Buffering: no` header is present in gateway responses

### Authentication errors

- Set `api_key` in the pipeline Valves to match one of the keys in `KNOWLEDGEHUB_API_KEYS`
- Or set the `KNOWLEDGEHUB_API_KEY` environment variable

### Fallback mode active (responses without RAG)

- This means the gateway was unreachable when the pipeline started
- Fix the gateway connection and update the Valves (this triggers a health re-check)
- Or restart Open WebUI after the gateway is healthy

## 7. Upgrading

When upgrading KnowledgeHub or the pipeline:

1. Pull the latest code:
   ```bash
   git pull origin main
   ```

2. Restart the stack:
   ```bash
   docker compose down
   docker compose up -d --build
   ```

3. If the pipeline file changed, re-upload it via the Admin UI or restart Open WebUI to pick up the volume-mounted changes.
