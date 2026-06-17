"""
Integration tests for ingestion + retrieval + grounded answering.

These exercise the real local embedding model, so they download a small model on
first run. If the model can't be loaded (e.g. offline CI), the tests skip cleanly.
The LLM is replaced with a deterministic fake so we test RAG wiring, not a model.
"""

from pathlib import Path

import pytest


SAMPLE_DOC = """\
Room 204 Projector Setup

To connect a laptop to the projector in Room 204, use the HDMI cable located in
the wall plate. Press the INPUT button on the projector remote and select HDMI 1.
If there is no signal, set the laptop display mode to Duplicate using Windows + P.

The Room 204 projector model is the Epson EB-2250U. Its default standby time is
ten minutes.
"""


@pytest.fixture()
def ingested_doc(tmp_path: Path, clean_store, embedder_available):
    if not embedder_available:
        pytest.skip("Embedding model unavailable (offline).")
    from backend.ingest import ingest_file

    doc = tmp_path / "room204.md"
    doc.write_text(SAMPLE_DOC, encoding="utf-8")
    result = ingest_file(doc, filename="room204.md")
    assert result.chunk_count >= 1
    return result


def test_ingestion_creates_chunks(ingested_doc, clean_store):
    docs = clean_store.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "room204.md"
    assert docs[0]["chunk_count"] == ingested_doc.chunk_count


def test_retrieval_finds_relevant_chunk(ingested_doc):
    from backend.rag import retrieve

    hits = retrieve("How do I connect a laptop to the Room 204 projector?")
    assert hits, "Expected at least one relevant chunk"
    assert "HDMI" in hits[0]["text"]
    assert hits[0]["score"] >= 0.25


def test_answer_uses_context_and_cites(ingested_doc, monkeypatch):
    """With relevant context, answer_question returns citations and used_context=True."""
    import backend.rag as rag

    class FakeLLM:
        name = "fake"
        model = "fake-model"

        def generate(self, system_prompt, messages):
            # Prove the context was injected into the system prompt.
            assert "HDMI" in system_prompt
            return "Use the HDMI cable and select HDMI 1 on the remote [1]."

    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM())

    resp = rag.answer_question("How do I connect a laptop in Room 204?")
    assert resp.used_context is True
    assert resp.citations
    assert resp.citations[0].filename == "room204.md"
    assert "HDMI" in resp.answer


def test_delete_removes_document(ingested_doc, clean_store):
    deleted = clean_store.delete_document(ingested_doc.document_id)
    assert deleted == ingested_doc.chunk_count
    assert clean_store.list_documents() == []
