# Architecture

## System Overview

KnowledgeHub is a **proxy-based RAG system** that sits between a chat frontend (Open WebUI) and an LLM backend (Ollama/vLLM). It intercepts every message, detects context, retrieves relevant knowledge, and enriches the LLM prompt — all transparently to the user.

```
┌─────────────┐       ┌─────────────────────────────────────────────────────┐       ┌──────────┐
│             │       │                   KnowledgeHub                     │       │          │
│  Open WebUI │──────>│  ┌─────────┐   ┌───────────┐   ┌──────────────┐   │──────>│  Ollama  │
│  (port 3000)│       │  │ Gateway │──>│ Detection │──>│ RAG Pipeline │   │       │  / vLLM  │
│             │<──────│  │  :8000  │   │  Engine   │   │              │   │<──────│          │
└─────────────┘       │  └────┬────┘   └─────┬─────┘   └──────┬───────┘   │       └──────────┘
                      │       │              │                │           │
                      │       v              v                v           │
                      │  ┌─────────┐   ┌───────────┐   ┌──────────────┐  │
                      │  │  Conv   │   │  Rule DB  │   │ Knowledge DB │  │
                      │  │ Manager │   │(detection │   │ + VectorStore│  │
                      │  └─────────┘   │  rules)   │   └──────────────┘  │
                      │                └───────────┘                     │
                      │                                                  │
                      │  ┌──────────────────────────────────────────┐    │
                      │  │       Admin Dashboard (:8001)            │    │
                      │  │  Rules | Contexts | Knowledge | Analytics│    │
                      │  └──────────────────────────────────────────┘    │
                      └─────────────────────────────────────────────────┘
```

---

## Component Diagram

### Gateway (port 8000)

The main entry point. Exposes an OpenAI-compatible API.

```
Gateway
├── routes/
│   ├── chat.py          # POST /v1/chat/completions, GET /v1/models
│   ├── health.py        # GET /health, GET /health/ready
│   ├── detection.py     # POST /api/v1/detection/detect
│   ├── knowledge.py     # CRUD /api/v1/knowledge/*
│   └── query.py         # POST /api/v1/query/ask
├── services/
│   ├── conversation.py  # ConversationManager (session tracking)
│   ├── openwebui_client.py  # Open WebUI API client
│   └── session_sync.py  # Session/user sync with Open WebUI
└── schemas/
    └── chat.py          # OpenAI-compatible Pydantic models
```

### Detection Engine

Evaluates rules in parallel, assigns contexts.

```
Detection
├── engine.py            # DetectionEngine orchestrator
│   ├── detect()         # Evaluate rules → DetectionResult
│   ├── detect_and_enrich()  # Detect + RAG + inject system message
│   └── execute_actions()    # Trigger actions (enrich, tag, log)
├── rules.py             # Rule ABC + implementations
│   ├── KeywordRule      # Exact/partial keyword matching
│   ├── RegexRule        # Regex pattern matching
│   ├── SemanticRule     # Cosine similarity to reference texts
│   ├── CompositeRule    # AND/OR/NOT combinations
│   └── LLMRule          # LLM-based classification
└── triggers.py          # Action system (enrich_prompt, tag, log)
```

### Knowledge Service

Manages the knowledge base: storage, retrieval, extraction.

```
Knowledge
├── service.py           # KnowledgeService orchestrator
│   ├── add_knowledge()  # Store + embed + vectorize
│   ├── search_knowledge()  # Semantic search + context filter
│   ├── build_rag_context() # Format results for LLM injection
│   ├── extract_knowledge_from_conversation()  # LLM extraction
│   ├── verify_knowledge()  # Admin approve/reject
│   └── bulk_import()    # Batch import with chunking
├── vectorstore.py       # VectorStore ABC
│   ├── ChromaVectorStore  # Mini profile
│   └── QdrantVectorStore  # Enterprise profile
└── embeddings.py        # EmbeddingProvider ABC
    ├── LocalEmbeddings    # sentence-transformers
    └── OllamaEmbeddings   # Ollama API
```

### LLM Abstraction

```
LLM
├── base.py              # LLMProvider ABC (complete, embed, list_models, health_check)
├── ollama.py            # OllamaProvider (retry, connection pooling)
├── vllm.py              # VLLMProvider (GPU, LoRA, batch)
├── factory.py           # Singleton factory
├── rag.py               # RAGOrchestrator
├── prompts.py           # Configurable prompt templates
└── models.py            # Pydantic models (ChatCompletion, etc.)
```

### Admin Dashboard (port 8001)

```
Admin
├── routes/
│   ├── rules.py         # CRUD + test + hot-reload
│   ├── contexts.py      # CRUD + hierarchy + stats
│   ├── knowledge.py     # List + verify/reject + import/export
│   ├── analytics.py     # Overview, trends, growth
│   └── dashboard.py     # HTML dashboard (HTMX + Tailwind)
├── schemas/             # Pydantic request/response models
└── dependencies.py      # API key authentication
```

---

## Data Flow

### Chat Request Flow

```
1. User sends message via Open WebUI
   │
2. Gateway receives POST /v1/chat/completions
   │
3. ConversationManager persists the message
   │
4. DetectionEngine.detect_and_enrich():
   │  a. Load rules from DB (keyword, regex, semantic, composite)
   │  b. Evaluate all rules in PARALLEL (asyncio.gather)
   │  c. Merge results → DetectionResult (topics, confidence)
   │  d. If topics detected:
   │     - KnowledgeService.build_rag_context(query, topics)
   │     - Inject system message with retrieved knowledge
   │
5. Forward enriched messages to LLM backend
   │
6. Return response (streaming SSE or JSON)
   │
7. Persist assistant response in DB
```

### Knowledge Extraction Flow

```
1. Admin triggers extraction for a conversation
   │
2. KnowledgeService.extract_knowledge_from_conversation():
   │  a. Load unprocessed messages
   │  b. Format as prompt for LLM
   │  c. LLM returns JSON array of facts
   │  d. Each fact → add_knowledge() (embed + store)
   │  e. Mark messages as extracted
   │
3. Items appear in admin pending queue
   │
4. Admin verifies/rejects each item
   │  - Verify → item stays in vector store
   │  - Reject → item removed from vector store
```

---

## Design Decisions

### Why a Proxy Architecture?

**Decision:** KnowledgeHub sits between frontend and LLM, rather than being embedded in either.

**Rationale:**
- Frontend-agnostic — works with any OpenAI-compatible client, not just Open WebUI
- LLM-agnostic — swap Ollama for vLLM without changing anything else
- Transparent — users don't need to learn a new interface
- Testable — each component can be tested independently

### Why Parallel Rule Evaluation?

**Decision:** All detection rules evaluate concurrently via `asyncio.gather`.

**Rationale:**
- Rules are I/O-bound (embedding lookups, LLM calls for SemanticRule/LLMRule)
- Parallel execution keeps detection under 100ms for keyword/regex rules
- Timeout per rule (default 5s) prevents slow rules from blocking others

### Why Two Database Strategies?

**Decision:** SQLite for mini profile, PostgreSQL for enterprise.

**Rationale:**
- SQLite requires zero setup — ideal for single-machine deployments
- PostgreSQL handles concurrency, replication, and larger datasets
- SQLAlchemy async abstraction makes the code identical for both
- Migration path is documented (see [Migration Guide](MIGRATION.md))

### Why Separate Vector Store from Relational DB?

**Decision:** Knowledge metadata in SQLite/PostgreSQL, embeddings in ChromaDB/Qdrant.

**Rationale:**
- Vector stores are optimized for approximate nearest neighbor search
- Relational DB handles structured queries (filter by context, verified status)
- Separation allows scaling the vector store independently

---

## Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Proxy architecture | Frontend/LLM agnostic | Extra network hop (~5ms latency) |
| SQLite for mini | Zero-config deployment | Single-writer concurrency limit |
| In-memory rule evaluation | Fast detection | Rules must fit in memory |
| LLM-based extraction | High quality knowledge | Slower, needs LLM call |
| Keyword rules first | Fast, predictable | Less nuanced than semantic |
| Unverified by default | Safety — admin reviews | Requires manual review effort |
| ChromaDB embedded mode | No external service | Limited scalability |

---

## Security Architecture

```
                    ┌─── Public Zone ───┐
                    │                   │
Internet ─── TLS ──┤  Open WebUI :3000 │
                    │                   │
                    └────────┬──────────┘
                             │ (internal network)
            ┌────────────────┼────────────────┐
            │                │                │
            │   Gateway :8000 (no auth*)      │
            │   Admin :8001 (API key auth)    │
            │   Ollama :11434 (internal only) │
            │   ChromaDB :8100 (internal only)│
            │                                 │
            └───── Docker Network (bridge) ───┘

* Gateway auth optional via API_KEYS setting
```

- Docker bridge network isolates internal services
- Only ports 3000 (frontend) and 8001 (admin) should be exposed externally
- Admin API requires `X-API-Key` header
- Non-root container user (`khub`) for production images
