"""
Configurable prompt templates for the KnowledgeHub RAG pipeline.

All prompts are defined as module-level constants and can be overridden
by passing custom templates to the :class:`RAGOrchestrator`.

Token estimation uses the ``~4 chars ≈ 1 token`` heuristic unless a
real tokenizer is configured.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# RAG system prompt
# ═══════════════════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """\
Sei un assistente aziendale esperto. Usa le seguenti informazioni dalla \
knowledge base per rispondere in modo accurato e completo. Se l'informazione \
richiesta non è presente nella knowledge base, dillo chiaramente e rispondi \
al meglio delle tue conoscenze.

=== KNOWLEDGE BASE ===
{knowledge}
=== FINE KNOWLEDGE BASE ===

Linee guida:
- Cita le fonti quando possibile (es. [Fonte: documento, Contesto: IT]).
- Non inventare informazioni non presenti nella knowledge base.
- Se più fonti sono rilevanti, sintetizzale in una risposta coerente.
- Rispondi nella stessa lingua dell'utente."""

# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge extraction prompt
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_EXTRACTION_PROMPT = """\
You are a knowledge extraction assistant. Analyse the assistant's response \
below in the context of the user's question and extract distinct, \
self-contained facts, procedures, or decisions that are worth saving in a \
corporate knowledge base.

Rules:
- Each extracted item MUST be a complete, stand-alone statement.
- Include specific details: names, numbers, dates, configurations.
- Omit greetings, chitchat, opinions, and filler.
- Do NOT extract information that merely rephrases the user's question.
- Return ONLY a JSON array of objects with "content" and "confidence" keys.
- "confidence" is a float 0.0–1.0 indicating how valuable the fact is.
- If nothing is worth extracting, return an empty array: []

Example output:
[
  {{"content": "The production database backup runs daily at 02:00 UTC.", "confidence": 0.9}},
  {{"content": "Maximum file upload size is 50MB.", "confidence": 0.7}}
]"""

KNOWLEDGE_EXTRACTION_USER_TEMPLATE = """\
User question:
{user_message}

Assistant response:
{assistant_response}

Detected contexts: {contexts}

Extract knowledge items as a JSON array:"""

# ═══════════════════════════════════════════════════════════════════════════════
# Reranking prompt
# ═══════════════════════════════════════════════════════════════════════════════

RERANK_PROMPT = """\
You are a relevance judge. Given the user's query and a list of knowledge \
base excerpts, rank them by relevance to the query.

Return ONLY a JSON array of indices (0-based) ordered from most to least \
relevant. Include only indices of excerpts that are actually relevant to \
the query. If none are relevant, return an empty array: []

User query: {query}

Excerpts:
{excerpts}

Ranked indices (JSON array):"""

# ═══════════════════════════════════════════════════════════════════════════════
# Summarization prompt
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARIZE_PROMPT = """\
Summarise the following text concisely, preserving all key facts, numbers, \
and technical details. The summary should be self-contained and useful as a \
knowledge base entry.

Text:
{text}

Summary:"""

# ═══════════════════════════════════════════════════════════════════════════════
# Source attribution templates
# ═══════════════════════════════════════════════════════════════════════════════

SOURCE_ATTRIBUTION_HEADER = "\n\n---\n**Fonti utilizzate:**\n"

SOURCE_ATTRIBUTION_ITEM = "- [{source_type}] {contexts} (rilevanza: {score:.0%})\n"

# ═══════════════════════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════════════════════

# Approximate chars-per-token ratio.  The default 4.0 is a conservative
# heuristic for English/Italian text.  Override with a real tokenizer
# for accuracy.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using a heuristic."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens*."""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
