"""
Central application configuration.

We use pydantic-settings so that every value can be overridden by an environment
variable (or a `.env` file) without touching the code. This is a production best
practice: secrets and environment-specific knobs never live in source control.

Import the singleton `settings` anywhere in the backend:

    from backend.config import settings
    print(settings.OPENAI_MODEL)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App / server ----
    APP_NAME: str = "Technical Services RAG Assistant"
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "http://localhost:8501,http://localhost:3000"

    # ---- Admin auth ----
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "change-me-please"

    # ---- Rate limiting ----
    RATE_LIMIT_CHAT: str = "20/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "60/minute"

    # ---- LLM provider ----
    LLM_PROVIDER: str = "auto"  # auto | openai | ollama
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 800

    # ---- Embeddings ----
    EMBEDDING_PROVIDER: str = "sentence_transformers"  # sentence_transformers | openai
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # ---- Vector store ----
    VECTOR_BACKEND: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "technical_services"

    # ---- Chunking / retrieval ----
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_TOP_K: int = 4
    MIN_RELEVANCE_SCORE: float = 0.25

    # ---- Uploads ----
    MAX_UPLOAD_MB: int = 25
    DATA_DIR: str = "./data"

    # -- Derived helpers ---------------------------------------------------

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse the comma-separated CORS string into a clean list."""
        raw = self.CORS_ORIGINS.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def openai_enabled(self) -> bool:
        """OpenAI is only available when a key is actually configured."""
        return bool(self.OPENAI_API_KEY.strip())

    @property
    def resolved_llm_provider(self) -> str:
        """
        Resolve "auto" into a concrete provider.

        Rule (per project spec): use OpenAI only if a key is provided,
        otherwise fall back to Ollama for free local inference.
        """
        if self.LLM_PROVIDER == "auto":
            return "openai" if self.openai_enabled else "ollama"
        return self.LLM_PROVIDER

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the .env file is parsed only once per process."""
    return Settings()


# Convenience singleton for simple imports.
settings = get_settings()
