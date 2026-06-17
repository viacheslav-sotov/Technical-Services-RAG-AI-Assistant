"""
Embedding providers (modular).

RAG concept
-----------
An *embedding* is a vector of numbers that captures the meaning of a piece of
text. Texts with similar meaning have vectors that are close together. We embed
every document chunk at ingestion time and embed the question at query time,
then compare them to find relevant context.

Two providers are supported and selected via `EMBEDDING_PROVIDER`:
  * sentence_transformers  -> local, free, no API key (default)
  * openai                 -> hosted, paid (text-embedding-3-small etc.)

The local model is lazily loaded on first use so importing this module is cheap
and tests stay fast.
"""

from __future__ import annotations

import abc
from functools import lru_cache
from typing import List, Optional

from backend.config import settings
from backend.utils import logger


class BaseEmbedder(abc.ABC):
    """Common interface for all embedding providers."""

    name: str = "base"
    dimension: Optional[int] = None

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        # By default a query is embedded exactly like a document.
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers (default, free)."""

    def __init__(self, model_name: str) -> None:
        # Import here so the heavy dependency only loads when this provider is used.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model '%s' ...", model_name)
        self._model = SentenceTransformer(model_name)
        self.name = f"sentence_transformers:{model_name}"
        # Newer sentence-transformers renamed this method; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = self._model.get_embedding_dimension()
        else:
            self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine-friendly
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """Hosted embeddings via the OpenAI API (paid, optional)."""

    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        if not settings.openai_enabled:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        self._client = OpenAI(**kwargs)
        self._model_name = model_name
        self.name = f"openai:{model_name}"
        # Known dimensions for the common small model; not strictly required.
        self.dimension = 1536 if "3-small" in model_name else None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in resp.data]


@lru_cache
def get_embedder() -> BaseEmbedder:
    """Return a cached embedder based on configuration."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        return OpenAIEmbedder(settings.OPENAI_EMBEDDING_MODEL)
    if provider in {"sentence_transformers", "st", "local"}:
        return SentenceTransformerEmbedder(settings.EMBEDDING_MODEL)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{provider}'.")
