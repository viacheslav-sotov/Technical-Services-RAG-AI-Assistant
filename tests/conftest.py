"""
Pytest configuration.

We set environment variables BEFORE importing any backend module so the cached
`settings` singleton points at an isolated, temporary vector store / data dir.
This keeps tests from touching your real chroma_db.
"""

from __future__ import annotations

import os
import tempfile

# --- Isolate state (must happen before backend imports) ---
_TMP = tempfile.mkdtemp(prefix="tsrag_tests_")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(_TMP, "chroma_db"))
os.environ.setdefault("DATA_DIR", os.path.join(_TMP, "data"))
os.environ.setdefault("CHROMA_COLLECTION", "test_collection")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "test-pass")
os.environ.setdefault("LLM_PROVIDER", "ollama")  # never calls out in unit tests
# Keep rate limits generous so tests don't trip them.
os.environ.setdefault("RATE_LIMIT_CHAT", "1000/minute")
os.environ.setdefault("RATE_LIMIT_UPLOAD", "1000/minute")
os.environ.setdefault("RATE_LIMIT_DEFAULT", "1000/minute")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def embedder_available() -> bool:
    """True if the local embedding model can be loaded (needs a one-time download)."""
    try:
        from backend.embeddings import get_embedder

        get_embedder().embed_query("warmup")
        return True
    except Exception:
        return False


@pytest.fixture()
def clean_store():
    """Provide a freshly-reset vector store for each test that needs one."""
    from backend.vector_store import ChromaVectorStore, get_vector_store

    store = get_vector_store()
    if isinstance(store, ChromaVectorStore):
        store.reset()
    yield store
    if isinstance(store, ChromaVectorStore):
        store.reset()
