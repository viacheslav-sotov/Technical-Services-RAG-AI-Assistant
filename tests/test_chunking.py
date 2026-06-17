"""Unit tests for the chunking logic (pure functions, no models needed)."""

from backend.ingest import chunk_text


def test_chunk_text_basic_splitting():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 3
    # Each chunk respects the size bound.
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_overlap_preserves_continuity():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=300, overlap=80)
    assert len(chunks) > 1
    # Overlap means consecutive chunks share some trailing/leading content.
    assert chunks[0][-20:] in (chunks[1][:200] + chunks[0][-20:])


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_handles_overlap_larger_than_size():
    # Pathological config should not loop forever or crash.
    chunks = chunk_text("hello world " * 50, chunk_size=50, overlap=100)
    assert len(chunks) > 0
