# User Guide

KnowledgeHub enhances your chat experience by automatically providing relevant company knowledge when you ask questions. You interact with it through **Open WebUI** — the same chat interface you already know.

---

## Getting Started

### 1. Access the Chat

Open your browser and navigate to:

```
http://localhost:3000
```

If authentication is enabled, log in with your credentials.

### 2. Start a Conversation

Type your question naturally. KnowledgeHub works behind the scenes:

```
You:  Come si configura il database per il progetto Alpha?
Bot:  Per configurare il database del progetto Alpha, segui questi passaggi...
      [Il bot include automaticamente informazioni dalla knowledge base interna]
```

### 3. Available Models

In Open WebUI, select a model from the dropdown. KnowledgeHub proxies all requests through its Gateway, so any model listed works with knowledge enrichment.

---

## How It Works

When you send a message, KnowledgeHub:

1. **Analyzes your question** — detects topics like "database", "onboarding", "project"
2. **Searches the knowledge base** — finds relevant internal documents and past answers
3. **Enriches your prompt** — injects the found knowledge into the LLM's context
4. **Returns an informed answer** — the LLM responds with grounded, company-specific information

You don't need to do anything special — the enrichment happens automatically.

---

## Features

### Context-Aware Responses

The system recognizes topics in your questions:

| You ask about... | KnowledgeHub enriches with... |
|---|---|
| Projects | Project documentation, milestones, team info |
| Procedures | Step-by-step internal procedures |
| Onboarding | New employee guides, first-day checklists |
| Database | Configuration guides, troubleshooting tips |
| Errors | Error code documentation, known fixes |

### Multi-Language Support

KnowledgeHub works with both Italian and English. Ask in whichever language is most natural:

```
"Qual è la procedura per richiedere ferie?"
"How do I request time off?"
```

### Conversation History

Your conversations are saved and can be accessed from the Open WebUI sidebar. The system learns from conversations — admins can extract useful knowledge from chat history.

---

## Best Practices

### Be Specific

More specific questions get better results:

```
Less effective:  "Dimmi del progetto"
More effective:  "Qual è lo stato attuale del progetto Alpha e chi è il project manager?"
```

### Include Context

Mention the topic area to help detection:

```
Less effective:  "Come si fa?"
More effective:  "Come si fa a configurare il connection pooling del database?"
```

### Ask Follow-Up Questions

KnowledgeHub tracks conversation context. Follow-up questions benefit from the already-detected topics:

```
You:  Come si configura PostgreSQL?
Bot:  [risposta con info dalla knowledge base]
You:  E per il connection pooling?
Bot:  [risposta con ulteriori dettagli, stesso contesto "database"]
```

### Report Incorrect Information

If the bot provides incorrect information, let your admin know. They can:
- Correct the knowledge base entry
- Reject inaccurate auto-extracted knowledge
- Add verified correct information

---

## What KnowledgeHub Does NOT Do

- **Does not access external internet** — all knowledge comes from the internal knowledge base
- **Does not store personal data** — conversations are linked to session IDs, not personal accounts (unless Open WebUI auth is configured)
- **Does not replace official documentation** — treat responses as helpful guidance, always verify critical procedures through official channels
- **Does not execute code or actions** — it only provides information

---

## FAQ

**Q: Why does the bot sometimes not include company-specific info?**
A: The detection rules may not match your question's topic. Try rephrasing with more specific keywords, or ask your admin to add relevant detection rules.

**Q: Can I upload documents for the bot to learn from?**
A: Document upload is available through the admin interface. Contact your admin to add documents to the knowledge base.

**Q: Are my conversations private?**
A: Conversations are stored for knowledge extraction purposes. Admins can review conversation history. Do not share sensitive personal information in the chat.

**Q: The bot is slow — what can I do?**
A: Response time depends on the LLM model and server load. Shorter questions typically get faster responses. If consistently slow, contact your admin.
