"""API tests: health check, validation, auth, and the upload->ingest->refuse flow."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "vector_store" in body
    assert "llm" in body


def test_documents_endpoint_shape():
    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert "total_chunks" in body


def test_chat_validation_rejects_empty_question():
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_admin_stats_requires_auth():
    # No credentials -> 401
    assert client.get("/admin/stats").status_code == 401
    # Wrong credentials -> 401
    bad = client.get("/admin/stats", auth=("admin", "wrong"))
    assert bad.status_code == 401
    # Correct credentials (set in conftest) -> 200
    ok = client.get("/admin/stats", auth=("admin", "test-pass"))
    assert ok.status_code == 200
    assert "total_chunks" in ok.json()


def test_upload_rejects_unsupported_type():
    files = {"file": ("notes.exe", b"binary", "application/octet-stream")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400


def test_chat_refuses_when_no_relevant_context(clean_store, embedder_available):
    """With an empty knowledge base, the assistant must refuse (no hallucination)."""
    if not embedder_available:
        pytest.skip("Embedding model unavailable (offline).")
    resp = client.post("/chat", json={"question": "What cable does Room 999 use?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_context"] is False
    assert "could not find" in body["answer"].lower()
    assert body["citations"] == []
