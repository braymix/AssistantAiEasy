# Admin Guide

The Admin Dashboard (port 8001) provides full management of KnowledgeHub's detection rules, contexts, knowledge base, and analytics.

All REST API endpoints require the `X-API-Key` header.

---

## 1. Managing Detection Rules

Rules determine which topics KnowledgeHub detects in user messages. When a rule matches, related knowledge is retrieved and injected into the LLM prompt.

### Rule Types

| Type | Description | Example |
|------|-------------|---------|
| **keyword** | Matches if any keyword is found in the text | `["database", "sql", "query"]` |
| **regex** | Matches via regular expression patterns | `"ERR-\d+"`, `"come (si fa\|faccio)"` |
| **semantic** | Matches by cosine similarity to reference texts | `"nuovo dipendente"` with threshold 0.75 |
| **composite** | Combines keyword + regex with AND/OR/NOT logic | Keywords OR regex pattern |

### Create a Rule

```bash
curl -X POST http://localhost:8001/api/v1/admin/rules \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "project_mention",
    "description": "Detects project-related questions",
    "rule_type": "keyword",
    "rule_config": {
      "keywords": ["progetto", "project", "milestone"],
      "case_sensitive": false
    },
    "target_contexts": ["projects"],
    "priority": 10,
    "enabled": true
  }'
```

### List Rules

```bash
# All rules
curl http://localhost:8001/api/v1/admin/rules \
  -H "X-API-Key: your-api-key"

# Filter by type
curl "http://localhost:8001/api/v1/admin/rules?rule_type=keyword&enabled=true" \
  -H "X-API-Key: your-api-key"
```

### Test a Rule

Test how a rule matches against sample text without saving:

```bash
curl -X POST http://localhost:8001/api/v1/admin/rules/{rule_id}/test \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Come faccio a configurare il database?"}'
```

### Hot-Reload Rules

After modifying rules, reload the detection engine without restart:

```bash
curl -X POST http://localhost:8001/api/v1/admin/rules/reload \
  -H "X-API-Key: your-api-key"
```

### Rule Priority

Rules with higher `priority` values are evaluated first. When multiple rules match, all their contexts are merged. Priority affects ordering in results but not whether a rule fires.

---

## 2. Managing Contexts

Contexts are topic categories that rules assign to detected conversations. They determine which knowledge base segments are searched during RAG.

### Create a Context

```bash
curl -X POST http://localhost:8001/api/v1/admin/contexts \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "onboarding",
    "description": "Informazioni per nuovi dipendenti",
    "parent_id": null
  }'
```

### Context Hierarchy

Contexts support parent-child relationships. A child context inherits the parent's knowledge scope:

```bash
# Create parent
curl -X POST .../contexts -d '{"name": "hr", "description": "Human Resources"}'

# Create child under HR
curl -X POST .../contexts -d '{"name": "onboarding", "parent_id": "<hr_context_id>"}'
```

### View Context Stats

```bash
curl http://localhost:8001/api/v1/admin/contexts/{context_id}/stats \
  -H "X-API-Key: your-api-key"
```

Returns: knowledge item count, rule count, mention count for that context.

---

## 3. Knowledge Review

Knowledge items can be added manually, imported in bulk, or extracted automatically from conversations. Extracted items are **unverified** by default and require admin approval.

### Review Pending Items

```bash
curl http://localhost:8001/api/v1/admin/knowledge/pending \
  -H "X-API-Key: your-api-key"
```

### Approve / Reject

```bash
# Approve
curl -X POST http://localhost:8001/api/v1/admin/knowledge/{item_id}/verify \
  -H "X-API-Key: your-api-key"

# Reject (removes from vector store)
curl -X POST http://localhost:8001/api/v1/admin/knowledge/{item_id}/reject \
  -H "X-API-Key: your-api-key"
```

### Bulk Import

Import multiple documents at once:

```bash
curl -X POST http://localhost:8001/api/v1/admin/knowledge/import \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Full text of document 1...", "source": "wiki"},
      {"content": "Full text of document 2...", "source": "confluence"}
    ],
    "contexts": ["procedures"],
    "chunk_size": 512,
    "chunk_overlap": 50
  }'
```

### Export Knowledge Base

```bash
# All items
curl http://localhost:8001/api/v1/admin/knowledge/export \
  -H "X-API-Key: your-api-key"

# Verified only
curl "http://localhost:8001/api/v1/admin/knowledge/export?verified_only=true" \
  -H "X-API-Key: your-api-key"
```

---

## 4. Analytics and Monitoring

### Overview Dashboard

```bash
curl http://localhost:8001/api/v1/admin/analytics/overview \
  -H "X-API-Key: your-api-key"
```

Returns: total rules (active/total), contexts, knowledge items (verified/pending), conversations, messages, documents.

### Context Usage

```bash
curl http://localhost:8001/api/v1/admin/analytics/contexts \
  -H "X-API-Key: your-api-key"
```

Returns per-context: knowledge count, rule count, mention count.

### Rule Performance

```bash
curl http://localhost:8001/api/v1/admin/analytics/rules \
  -H "X-API-Key: your-api-key"
```

Returns per-rule: type, enabled status, priority, target contexts.

### Conversation Trends

```bash
# Last 30 days
curl "http://localhost:8001/api/v1/admin/analytics/conversations?period_days=30" \
  -H "X-API-Key: your-api-key"
```

Returns: total conversations in period + daily breakdown.

### Knowledge Growth

```bash
curl http://localhost:8001/api/v1/admin/analytics/knowledge \
  -H "X-API-Key: your-api-key"
```

Returns: total/verified/pending counts, breakdown by content type and source.

---

## 5. Backup and Restore

### Create a Backup

```bash
# Full backup (DB + vector store)
python scripts/backup.py

# Database only
python scripts/backup.py --db-only

# Custom output directory
python scripts/backup.py --output /mnt/backups
```

### Retention Policy

Old backups are auto-deleted based on retention:

```bash
python scripts/backup.py --retention-days 30
```

### Restore from Backup

For SQLite, copy the backup file:

```bash
# Stop services
make down

# Restore
cp data/backups/knowledgehub_backup_db_20240115_120000.sqlite.gz /tmp/
gunzip /tmp/knowledgehub_backup_db_20240115_120000.sqlite.gz
cp /tmp/knowledgehub_backup_db_20240115_120000.sqlite data/sqlite/knowledgehub.db

# Restart
make up
```

For PostgreSQL:

```bash
gunzip -c data/backups/knowledgehub_backup_db_*.sql.gz | psql -h localhost -U knowledgehub -d knowledgehub
```

### Health Monitoring

Run the health check script periodically (e.g., via cron):

```bash
python scripts/health_check.py --json
```

Exit code 0 = all healthy, 1 = one or more services down.

---

## Best Practices

1. **Start with keyword rules** — they are fast and predictable. Add regex/semantic only when needed.
2. **Review extracted knowledge regularly** — the pending queue grows if left unchecked.
3. **Use context hierarchies** — group related topics under a parent for better organization.
4. **Test rules before enabling** — use the `/test` endpoint to validate matching behavior.
5. **Monitor analytics weekly** — identify unused rules and under-populated contexts.
6. **Schedule backups** — run `python scripts/backup.py` via cron daily.
