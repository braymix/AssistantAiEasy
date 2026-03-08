"""
Dual-mode configuration for KnowledgeHub.

Profiles:
  - mini:  Single-machine / mini-PC deployment (SQLite + Chroma + Ollama)
  - full:  Enterprise deployment (PostgreSQL + Qdrant + vLLM with GPU)

Usage:
    from src.config import get_settings
    settings = get_settings()
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Profile(str, Enum):
    MINI = "mini"
    FULL = "full"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"


class EmbeddingBackend(str, Enum):
    LOCAL = "local"
    OLLAMA = "ollama"


# ---------------------------------------------------------------------------
# Profile-specific defaults
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: dict[str, dict] = {
    "mini": {
        "database_url": "sqlite+aiosqlite:///./data/sqlite/knowledgehub.db",
        "vectorstore_backend": VectorStoreBackend.CHROMA,
        "llm_backend": LLMBackend.OLLAMA,
        "embedding_backend": EmbeddingBackend.LOCAL,
        "ollama_model": "llama3.2:3b",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "embedding_batch_size": 64,
        "max_concurrent_requests": 4,
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "full": {
        "database_url": "postgresql+asyncpg://knowledgehub:changeme@postgres:5432/knowledgehub",
        "vectorstore_backend": VectorStoreBackend.QDRANT,
        "llm_backend": LLMBackend.VLLM,
        "embedding_backend": EmbeddingBackend.OLLAMA,
        "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama_embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "embedding_batch_size": 128,
        "max_concurrent_requests": 32,
        "chunk_size": 1024,
        "chunk_overlap": 100,
    },
}


class Settings(BaseSettings):
    """KnowledgeHub application settings with dual-mode support."""

    model_config = SettingsConfigDict(
        env_prefix="KNOWLEDGEHUB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Profile ───────────────────────────────────────────────────────────
    profile: Profile = Profile.MINI
    debug: bool = False

    # ── Service ───────────────────────────────────────────────────────────
    app_name: str = "KnowledgeHub"
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    admin_port: int = 8001
    secret_key: str = "change-this-to-a-random-secret-key"
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = Field(default_factory=list)
    max_concurrent_requests: int = 4

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = ""
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # ── Vector Store ──────────────────────────────────────────────────────
    vectorstore_backend: VectorStoreBackend = VectorStoreBackend.CHROMA

    # Chroma
    chroma_host: str = "chroma"
    chroma_port: int = 8100
    chroma_collection: str = "knowledgehub"
    chroma_persist_dir: str = "./data/chroma"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "knowledgehub"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_backend: EmbeddingBackend = EmbeddingBackend.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    ollama_embedding_model: str = "nomic-embed-text"

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_backend: LLMBackend = LLMBackend.OLLAMA

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 120

    # vLLM
    vllm_base_url: str = "http://vllm:8000"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_max_tokens: int = 2048

    # ── Knowledge processing ──────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── Open WebUI integration ─────────────────────────────────────────────
    openwebui_url: str = "http://localhost:3000"
    openwebui_token: str = ""

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    log_format: Literal["json", "console"] = "json"

    @model_validator(mode="before")
    @classmethod
    def apply_profile_defaults(cls, values: dict) -> dict:
        """Merge profile-specific defaults for any field not explicitly set."""
        profile = values.get("profile") or values.get("KNOWLEDGEHUB_PROFILE", "mini")
        if isinstance(profile, str):
            profile = profile.lower()
        defaults = _PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["mini"])

        for key, default_value in defaults.items():
            env_key = f"KNOWLEDGEHUB_{key.upper()}"
            # Only apply default if the value was not explicitly provided
            if key not in values and env_key not in values:
                values[key] = default_value

        return values

    @property
    def is_mini(self) -> bool:
        return self.profile == Profile.MINI

    @property
    def is_full(self) -> bool:
        return self.profile == Profile.FULL

    @property
    def is_sqlite(self) -> bool:
        return "sqlite" in self.database_url

    @property
    def is_postgres(self) -> bool:
        return "postgresql" in self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
