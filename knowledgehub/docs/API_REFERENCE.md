# API Reference

KnowledgeHub exposes two APIs:

- **Gateway API** (port 8000) — OpenAI-compatible chat proxy + knowledge/detection endpoints
- **Admin API** (port 8001) — management REST API, requires `X-API-Key` header

---

## Gateway API

### Chat Completions

OpenAI-compatible endpoint. Open WebUI sends requests here.

```
POST /v1/chat/completions
```

**Request:**

```json
{
  "model": "phi3:mini",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Come si configura il database?"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": false,
  "user": "session-id-123"
}
```

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "phi3:mini",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Per configurare il database..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

**Response (streaming):** Server-Sent Events, each line is `data: <json>`, ending with `data: [DONE]`.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

### List Models

```
GET /v1/models
```

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [{"id": "phi3:mini", "object": "model", "owned_by": "knowledgehub"}]
}
```

### Health Check

```
GET /health
```

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy", "profile": "mini", "llm_backend": "ollama", "vectorstore": "chroma", "version": "0.1.0"}
```

### Readiness Probe

```
GET /health/ready
```

```json
{"status": "ready", "checks": {"llm": true}}
```

### Context Detection

```
POST /api/v1/detection/detect
```

```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Database query gave ERR-42", "context": {}}'
```

```json
{
  "triggered_rules": [
    {"rule_id": "...", "rule_name": "database_issues", "confidence": 0.67, "matched_keywords": ["database", "query"]},
    {"rule_id": "...", "rule_name": "error_codes", "confidence": 0.8, "matched_keywords": ["ERR-42"]}
  ],
  "suggested_topics": ["database", "errors"],
  "confidence": 0.74,
  "processing_time_ms": 12
}
```

### Knowledge Search

```
POST /api/v1/knowledge/search
```

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "connection pooling", "top_k": 5, "contexts": ["database"], "min_score": 0.7}'
```

### Add Knowledge

```
POST /api/v1/knowledge/add
```

```bash
curl -X POST http://localhost:8000/api/v1/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "PostgreSQL supports connection pooling via pgbouncer",
    "content_type": "manual",
    "contexts": ["database"],
    "created_by": "admin"
  }'
```

### Document CRUD

```
POST   /api/v1/knowledge/documents          # Create document
GET    /api/v1/knowledge/documents          # List documents
GET    /api/v1/knowledge/documents/{id}     # Get document
DELETE /api/v1/knowledge/documents/{id}     # Delete document
POST   /api/v1/knowledge/documents/upload   # Upload file
```

```bash
# Create document
curl -X POST http://localhost:8000/api/v1/knowledge/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "DB Guide", "content": "Full guide text..."}'

# List documents
curl "http://localhost:8000/api/v1/knowledge/documents?skip=0&limit=20"
```

### Query (Q&A)

```
POST /api/v1/query/ask
```

```bash
curl -X POST http://localhost:8000/api/v1/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I configure PostgreSQL?", "top_k": 5}'
```

```json
{"answer": "To configure PostgreSQL...", "sources": [{"content": "...", "score": 0.92}]}
```

---

## Admin API

All admin endpoints are prefixed with `/api/v1/admin` and require authentication.

```bash
# Set your API key
export API_KEY="your-admin-api-key"
```

### Rules

```
GET    /api/v1/admin/rules                     # List rules
POST   /api/v1/admin/rules                     # Create rule
GET    /api/v1/admin/rules/{id}                # Get rule
PUT    /api/v1/admin/rules/{id}                # Update rule
DELETE /api/v1/admin/rules/{id}                # Delete rule
POST   /api/v1/admin/rules/{id}/test           # Test rule
POST   /api/v1/admin/rules/reload              # Hot-reload
```

**Create rule:**

```bash
curl -X POST http://localhost:8001/api/v1/admin/rules \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "onboarding_topic",
    "rule_type": "semantic",
    "rule_config": {
      "reference_texts": ["nuovo dipendente", "primo giorno"],
      "threshold": 0.75
    },
    "target_contexts": ["onboarding"],
    "priority": 8
  }'
```

**Query parameters for list:**

| Param | Type | Description |
|-------|------|-------------|
| `rule_type` | string | Filter by type: keyword, regex, semantic, composite |
| `enabled` | bool | Filter by enabled status |
| `search` | string | Search in name/description |
| `limit` | int | Page size (default 50) |
| `offset` | int | Pagination offset |

### Contexts

```
GET    /api/v1/admin/contexts                         # List contexts
POST   /api/v1/admin/contexts                         # Create context
PUT    /api/v1/admin/contexts/{id}                    # Update context
DELETE /api/v1/admin/contexts/{id}                    # Delete context
GET    /api/v1/admin/contexts/{id}/knowledge          # Context knowledge items
GET    /api/v1/admin/contexts/{id}/stats              # Context statistics
```

**Query parameters for list:**

| Param | Type | Description |
|-------|------|-------------|
| `flat` | bool | Flat list (true) or hierarchical tree (false) |
| `search` | string | Search in name/description |

### Knowledge

```
GET    /api/v1/admin/knowledge                        # List items
GET    /api/v1/admin/knowledge/pending                # Pending review
POST   /api/v1/admin/knowledge/{id}/verify            # Approve
POST   /api/v1/admin/knowledge/{id}/reject            # Reject
DELETE /api/v1/admin/knowledge/{id}                   # Delete
POST   /api/v1/admin/knowledge/import                 # Bulk import
GET    /api/v1/admin/knowledge/export                 # Export all
```

**Query parameters for list:**

| Param | Type | Description |
|-------|------|-------------|
| `verified` | bool | Filter by verification status |
| `content_type` | string | Filter: manual, conversation_extract, document |
| `context` | string | Filter by context name |
| `search` | string | Full-text search in content |

### Analytics

```
GET /api/v1/admin/analytics/overview              # System-wide stats
GET /api/v1/admin/analytics/rules                 # Rule performance
GET /api/v1/admin/analytics/contexts              # Context usage
GET /api/v1/admin/analytics/conversations         # Conversation trends
GET /api/v1/admin/analytics/knowledge             # Knowledge growth
```

**Example — Overview:**

```bash
curl http://localhost:8001/api/v1/admin/analytics/overview \
  -H "X-API-Key: $API_KEY"
```

```json
{
  "data": {
    "total_rules": 8,
    "active_rules": 7,
    "total_contexts": 9,
    "total_knowledge_items": 150,
    "verified_knowledge_items": 120,
    "pending_knowledge_items": 30,
    "total_conversations": 500,
    "total_messages": 2400,
    "total_documents": 15
  }
}
```

---

## Response Envelope

All admin API responses are wrapped in an `ApiResponse` envelope:

```json
{
  "data": <response_object>,
  "meta": {
    "total": 100,
    "limit": 20,
    "offset": 0
  }
}
```

`meta` is included only for paginated list endpoints.

---

## Error Responses

```json
{"detail": "Resource not found"}          // 404
{"detail": "Invalid or missing API key"}  // 401
{"detail": "Bad request"}                 // 400
{"detail": "Internal server error"}       // 500
```

---

## Authentication

The Admin API uses API key authentication:

```
X-API-Key: your-api-key
```

Configure via the `ADMIN_API_KEY` environment variable. Multiple keys can be set with `API_KEYS=key1,key2`.

The Gateway API is unauthenticated by default. To require authentication, set `API_KEYS` in your `.env`.
