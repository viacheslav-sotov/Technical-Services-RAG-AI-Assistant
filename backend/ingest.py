"""
Document ingestion: parse -> chunk -> embed -> store.

RAG concept
-----------
LLMs have a limited context window and retrieval works best on focused passages,
so we split each document into overlapping *chunks*. Overlap preserves context
across chunk boundaries (a sentence cut in half is still recoverable). Each chunk
carries metadata (filename, page, chunk index) so we can cite it later.

Supported formats: PDF (pypdf), DOCX (python-docx), TXT, Markdown.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.config import settings
from backend.embeddings import get_embedder
from backend.utils import logger, new_id, safe_extension
from backend.vector_store import get_vector_store

# A "page segment" is (page_number_or_None, text). Plain-text formats have no pages.
PageSegment = Tuple[Optional[int], str]


@dataclass
class IngestResult:
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    char_count: int
    pages: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_count": self.chunk_count,
            "char_count": self.char_count,
            "pages": self.pages,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> List[PageSegment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments: List[PageSegment] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            segments.append((i, text))
    return segments


def _extract_docx(path: Path) -> List[PageSegment]:
    import docx  # python-docx

    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # python-docx has no reliable page concept; treat the whole doc as one segment.
    return [(None, "\n".join(paragraphs))] if paragraphs else []


def _extract_text(path: Path) -> List[PageSegment]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, text)] if text.strip() else []


def extract_segments(path: Path) -> List[PageSegment]:
    """Dispatch to the right parser based on file extension."""
    ext = safe_extension(path.name)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext in {".txt", ".md", ".markdown"}:
        return _extract_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    # Collapse excessive blank lines / trailing spaces while keeping structure.
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """
    Character-based sliding-window chunking with overlap.

    Character chunking is simple, deterministic, and dependency-free, which keeps
    behavior predictable for a portfolio project. (tiktoken is included in the
    requirements if you later want token-accurate budgeting.)
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP
    text = _normalize_whitespace(text)
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4  # guard against pathological config

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Try to break on a natural boundary (paragraph/sentence/space) near `end`.
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:  # only break if reasonably far in
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest_file(
    path: str | Path,
    filename: Optional[str] = None,
    document_id: Optional[str] = None,
) -> IngestResult:
    """
    Full pipeline for one file:
      1. parse to page segments
      2. chunk each segment (preserving page numbers)
      3. embed all chunks
      4. store vectors + metadata in the vector store
    """
    path = Path(path)
    filename = filename or path.name
    file_type = safe_extension(filename).lstrip(".")
    document_id = document_id or new_id("doc")
    uploaded_at = _dt.datetime.utcnow().isoformat() + "Z"

    segments = extract_segments(path)
    if not segments:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "If this is a scanned PDF, OCR is required (not included)."
        )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []
    char_count = 0
    chunk_index = 0
    page_numbers = {p for p, _ in segments if p is not None}

    for page_number, seg_text in segments:
        for chunk in chunk_text(seg_text):
            char_count += len(chunk)
            ids.append(f"{document_id}::chunk_{chunk_index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": file_type,
                    "page": page_number if page_number is not None else -1,
                    "chunk_index": chunk_index,
                    "uploaded_at": uploaded_at,
                }
            )
            chunk_index += 1

    if not documents:
        raise ValueError(f"Document '{filename}' produced no chunks.")

    logger.info("Embedding %d chunks from '%s' ...", len(documents), filename)
    embedder = get_embedder()
    embeddings = embedder.embed_documents(documents)

    store = get_vector_store()
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info("Ingested '%s' as document_id=%s (%d chunks)", filename, document_id, len(documents))

    return IngestResult(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        chunk_count=len(documents),
        char_count=char_count,
        pages=len(page_numbers) if page_numbers else None,
    )
