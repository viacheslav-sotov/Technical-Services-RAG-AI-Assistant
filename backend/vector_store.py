"""
Vector store abstraction.

RAG concept
-----------
A *vector store* holds the numeric embeddings of every document chunk. At query
time we embed the user's question and ask the store for the chunks whose vectors
are closest (most semantically similar). Those chunks become the "context" we
feed to the LLM.

This module defines a small `BaseVectorStore` interface so the rest of the app
never talks to ChromaDB directly. To switch to Pinecone / Supabase Vector later,
implement the same interface in a new class and select it in `get_vector_store()`.
"""

from __future__ import annotations

import abc
from collections import defaultdict
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.utils import ensure_dir, logger


class BaseVectorStore(abc.ABC):
    """Minimal contract every vector backend must satisfy."""

    @abc.abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        ...

    @abc.abstractmethod
    def query(
        self, embedding: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Return a list of {text, metadata, score} ordered most-relevant first."""

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document_id. Returns number deleted."""

    @abc.abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def count(self) -> int:
        ...

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        ...


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed store, persisted to disk for local development.

    We configure the collection to use cosine space so that distances map
    cleanly to a 0..1 similarity score (similarity = 1 - cosine_distance),
    which we use to decide whether retrieved context is relevant enough.
    """

    def __init__(self) -> None:
        persist_dir = str(ensure_dir(settings.CHROMA_PERSIST_DIR))
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready (dir=%s, collection=%s, count=%d)",
            persist_dir,
            settings.CHROMA_COLLECTION,
            self.count(),
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        # Chroma upserts in batches; keep batches modest to bound memory.
        batch = 256
        for i in range(0, len(ids), batch):
            self._collection.add(
                ids=ids[i : i + batch],
                embeddings=embeddings[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            # cosine distance -> similarity score in [0, 1] (clamped)
            score = max(0.0, 1.0 - float(dist))
            hits.append({"text": text, "metadata": meta or {}, "score": score})
        return hits

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregate chunk-level metadata back up to one row per document."""
        data = self._collection.get(include=["metadatas"])
        metas = data.get("metadatas", []) or []

        grouped: Dict[str, Dict[str, Any]] = {}
        chunk_counts: Dict[str, int] = defaultdict(int)
        for meta in metas:
            if not meta:
                continue
            doc_id = meta.get("document_id", "unknown")
            chunk_counts[doc_id] += 1
            if doc_id not in grouped:
                grouped[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "file_type": meta.get("file_type", ""),
                    "uploaded_at": meta.get("uploaded_at", ""),
                }
        for doc_id, info in grouped.items():
            info["chunk_count"] = chunk_counts[doc_id]
        return sorted(grouped.values(), key=lambda d: d.get("filename", ""))

    def count(self) -> int:
        return self._collection.count()

    def health(self) -> Dict[str, Any]:
        try:
            return {
                "backend": "chroma",
                "status": "ok",
                "collection": settings.CHROMA_COLLECTION,
                "chunk_count": self.count(),
                "persist_dir": settings.CHROMA_PERSIST_DIR,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Vector store health check failed")
            return {"backend": "chroma", "status": "error", "error": str(exc)}

    def reset(self) -> None:
        """Danger: wipe the collection. Used by tests."""
        self._client.delete_collection(settings.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


_store: Optional[BaseVectorStore] = None


def get_vector_store() -> BaseVectorStore:
    """
    Return a process-wide singleton store.

    Swap backends here when adding Pinecone/Supabase: read settings.VECTOR_BACKEND
    and return the matching implementation.
    """
    global _store
    if _store is None:
        backend = settings.VECTOR_BACKEND.lower()
        if backend == "chroma":
            _store = ChromaVectorStore()
        else:
            raise ValueError(
                f"Unsupported VECTOR_BACKEND '{backend}'. "
                "Implement BaseVectorStore and register it in get_vector_store()."
            )
    return _store
