# Proxy Mode – Open WebUI without Pipelines

If you prefer not to use Pipelines, KnowledgeHub Gateway can act as a **transparent proxy** between Open WebUI and Ollama/vLLM. Open WebUI sends requests to the Gateway (thinking it's Ollama), and the Gateway enriches them with RAG before forwarding to the real LLM backend.

> For the Pipeline-based approach, see [OPEN_WEBUI_SETUP.md](./OPEN_WEBUI_SETUP.md).

---

## How It Works

```
┌──────────────┐         ┌──────────────────────┐         ┌─────────┐
│  Open WebUI  │──chat──▶│  KnowledgeHub        │──chat──▶│  Ollama  │
│  :3000       │◀────────│  Gateway (:8000)     │◀────────│  :11434  │
│              │         │                      │         │          │
│  Thinks it's │         │  1. Detect context   │         │  (real   │
│  talking to  │         │  2. RAG enrichment   │         │  LLM)    │
│  "Ollama"    │         │  3. Forward to LLM   │         │          │
└──────────────┘         └──────────────────────┘         └─────────┘
```

Open WebUI is configured to point its **Ollama Base URL** (or OpenAI API URL) at the KnowledgeHub Gateway instead of directly at Ollama. The Gateway exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) that Open WebUI already knows how to call.

## Advantages

- **Zero pipeline installation** — no files to copy or upload
- **Works with any Open WebUI version** (0.1+)
- **Simpler architecture** — fewer moving parts
- **All RAG logic stays in the Gateway** — single point of configuration

## Disadvantages

- **Less UI integration** — no Valves configuration in Open WebUI admin
- **All-or-nothing** — every chat goes through the Gateway (no per-model toggle)
- **No fallback** — if the Gateway is down, Open WebUI cannot reach Ollama at all

## Setup

### Option 1: Environment Variable

Set `OLLAMA_BASE_URL` to point at the KnowledgeHub Gateway:

```bash
docker run -d \
  --name open-webui \
  --network knowledgehub_knowledgehub \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://gateway:8000 \
  -v open-webui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

### Option 2: OpenAI-Compatible API Mode

Open WebUI also supports connecting to OpenAI-compatible APIs. Since the Gateway exposes `/v1/chat/completions` and `/v1/models`, you can use this mode:

```bash
docker run -d \
  --name open-webui \
  --network knowledgehub_knowledgehub \
  -p 3000:8080 \
  -e ENABLE_OPENAI_API=true \
  -e OPENAI_API_BASE_URL=http://gateway:8000/v1 \
  -e OPENAI_API_KEY=knowledgehub \
  -v open-webui-data:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

> **Note**: The `OPENAI_API_KEY` value can be anything — the Gateway does not require auth on the chat endpoint by default. If you've configured API keys in KnowledgeHub, use one of those.

### Option 3: Docker Compose

```yaml
# docker-compose.openwebui.yml
version: "3.9"

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      # Point directly at KnowledgeHub Gateway
      - OLLAMA_BASE_URL=http://gateway:8000
      # Or use OpenAI-compatible mode:
      # - ENABLE_OPENAI_API=true
      # - OPENAI_API_BASE_URL=http://gateway:8000/v1
      # - OPENAI_API_KEY=knowledgehub
    volumes:
      - open-webui-data:/app/backend/data
    depends_on:
      - gateway
    restart: unless-stopped
    networks:
      - knowledgehub

volumes:
  open-webui-data:

networks:
  knowledgehub:
    external: true
```

### Option 4: Open WebUI Admin Settings

If Open WebUI is already running:

1. Open http://localhost:3000
2. Go to **Admin Panel** → **Settings** → **Connections**
3. Under **Ollama API**, change the URL to:
   ```
   http://gateway:8000
   ```
   (or `http://localhost:8000` if not using Docker networking)
4. Click **Save**

For OpenAI-compatible mode:
1. Go to **Admin Panel** → **Settings** → **Connections**
2. Under **OpenAI API**, set:
   - **API Base URL**: `http://gateway:8000/v1`
   - **API Key**: `knowledgehub` (or your configured key)
3. Click **Save**

## Verify

### Check the Gateway is proxying correctly

```bash
# List available models (should return KnowledgeHub's model)
curl http://localhost:8000/v1/models

# Test a chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Check Open WebUI sees the model

1. Open http://localhost:3000
2. In the model selector dropdown, you should see the KnowledgeHub model (e.g., `llama3.2:3b`)
3. Start a chat — the request will go through the Gateway with RAG enrichment

## Hybrid Mode

You can run **both** Proxy Mode and Pipeline Mode simultaneously:

1. Configure Open WebUI to connect directly to Ollama (normal setup)
2. Install the KnowledgeHub Pipeline for RAG-enriched conversations
3. Users choose which model/pipeline to use per conversation

This gives maximum flexibility: direct Ollama access for general chat, and KnowledgeHub-enriched responses when needed.

## Troubleshooting

### Open WebUI shows "No models available"

The Gateway's `/v1/models` endpoint returns the configured model. Verify:
```bash
curl http://localhost:8000/v1/models
```

If it returns an empty list, check that `KNOWLEDGEHUB_OLLAMA_MODEL` or `KNOWLEDGEHUB_VLLM_MODEL` is set correctly.

### Responses are slow

The Gateway adds detection + RAG lookup time on top of the LLM inference. Check:
- Gateway health: `curl http://localhost:8000/health/ready`
- Vector store connectivity
- Detection rules (simpler rules = faster detection)

### Streaming breaks

Ensure no reverse proxy between Open WebUI and the Gateway is buffering responses. If using nginx:

```nginx
location / {
    proxy_pass http://gateway:8000;
    proxy_buffering off;
    proxy_set_header X-Accel-Buffering no;
}
```

### Want to bypass RAG for some messages

In Proxy Mode, all messages go through the Gateway's detection engine. If no contexts are detected, the message is forwarded to the LLM without RAG enrichment — so effectively, unrelated messages are already "bypassed."

To completely skip KnowledgeHub for certain models, use the Pipeline approach instead, which gives per-model control.
