"""
FastAPI application: wires together ingestion, retrieval, generation, auth,
rate limiting, CORS, logging, and health checks.

Run locally:
    uvicorn backend.main:app --reload --port 8000

API routes:
    GET    /health                 -> liveness + component status
    POST   /upload                 -> upload a document (returns upload_id)
    POST   /ingest                 -> parse/chunk/embed/store an uploaded file
    POST   /chat                   -> ask a question (RAG)
    GET    /documents              -> list ingested documents
    DELETE /documents/{document_id}-> remove a document's chunks (admin)
    GET    /admin/stats            -> system stats (admin)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.auth import require_admin
from backend.config import settings
from backend.ingest import ingest_file
from backend.llm_provider import LLMError, get_llm
from backend.rag import answer_question
from backend.rate_limit import limiter
from backend.utils import (
    ensure_dir,
    file_sha256,
    is_allowed_file,
    logger,
    new_id,
    safe_extension,
)
from backend.vector_store import get_vector_store

# In-memory registry of uploaded-but-not-yet-ingested files.
# Maps upload_id -> {"path", "filename", "sha256"}. Fine for a single-process app.
_PENDING_UPLOADS: Dict[str, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Request / response schemas (input validation lives here)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)


class IngestRequest(BaseModel):
    upload_id: str = Field(min_length=1, max_length=128)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up persistent components so the first request isn't slow / surprising.
    ensure_dir(settings.DATA_DIR)
    ensure_dir(settings.CHROMA_PERSIST_DIR)
    get_vector_store()
    logger.info(
        "Startup complete | env=%s | llm=%s | embeddings=%s | vector=%s",
        settings.ENVIRONMENT,
        settings.resolved_llm_provider,
        settings.EMBEDDING_PROVIDER,
        settings.VECTOR_BACKEND,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Retrieval-Augmented Generation assistant for Technical Services / AV staff.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Rate limiting ----
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Catch-all error handler so clients always get clean JSON ----
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def health(request: Request):
    """Liveness + lightweight component status for monitors and the admin page."""
    store = get_vector_store()
    try:
        llm_health = get_llm().health()
    except Exception as exc:  # provider might be misconfigured; report, don't crash
        llm_health = {"provider": settings.resolved_llm_provider, "status": "error", "error": str(exc)}
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "llm": llm_health,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL
        if settings.EMBEDDING_PROVIDER == "sentence_transformers"
        else settings.OPENAI_EMBEDDING_MODEL,
        "vector_store": store.health(),
    }


@app.post("/upload", tags=["documents"])
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload(request: Request, file: UploadFile = File(...)):
    """
    Accept a document, validate it, and stage it on disk.

    Does NOT embed yet -- call /ingest with the returned upload_id to run the
    (potentially slow) embedding pipeline. Separating these keeps the upload
    request fast and lets the UI show progress.
    """
    filename = file.filename or "upload"
    if not is_allowed_file(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: pdf, txt, docx, md.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_UPLOAD_MB} MB limit.",
        )

    upload_id = new_id("up")
    stored_name = f"{upload_id}{safe_extension(filename)}"
    dest = ensure_dir(settings.DATA_DIR) / stored_name
    dest.write_bytes(contents)

    _PENDING_UPLOADS[upload_id] = {
        "path": str(dest),
        "filename": filename,
        "sha256": file_sha256(contents),
    }
    logger.info("Uploaded '%s' -> %s (%d bytes)", filename, upload_id, len(contents))
    return {
        "upload_id": upload_id,
        "filename": filename,
        "size_bytes": len(contents),
        "message": "Upload received. Call /ingest with this upload_id to index it.",
    }


@app.post("/ingest", tags=["documents"])
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
def ingest(request: Request, body: IngestRequest):
    """Parse, chunk, embed, and store a previously uploaded file."""
    pending = _PENDING_UPLOADS.get(body.upload_id)
    if not pending:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown upload_id. Upload the file again.",
        )
    try:
        result = ingest_file(pending["path"], filename=pending["filename"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for %s", body.upload_id)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    # Once indexed, drop it from the pending registry.
    _PENDING_UPLOADS.pop(body.upload_id, None)
    return result.as_dict()


@app.post("/chat", tags=["chat"])
@limiter.limit(settings.RATE_LIMIT_CHAT)
def chat(request: Request, body: ChatRequest):
    """Answer a question using RAG. Returns the answer plus source citations."""
    history = [m.model_dump() for m in body.history]
    try:
        response = answer_question(body.question, history=history)
    except LLMError as exc:
        # The retriever worked but the model backend is unreachable/misconfigured.
        raise HTTPException(status_code=502, detail=str(exc))
    return response.as_dict()


@app.get("/documents", tags=["documents"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def list_documents(request: Request):
    """List all ingested documents with chunk counts."""
    store = get_vector_store()
    return {"documents": store.list_documents(), "total_chunks": store.count()}


@app.delete("/documents/{document_id}", tags=["documents"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def delete_document(
    request: Request, document_id: str, _user: str = Depends(require_admin)
):
    """Delete all chunks for a document. Protected by admin Basic-Auth."""
    store = get_vector_store()
    deleted = store.delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    logger.info("Deleted document %s (%d chunks)", document_id, deleted)
    return {"document_id": document_id, "deleted_chunks": deleted}


@app.get("/admin/stats", tags=["admin"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def admin_stats(request: Request, _user: str = Depends(require_admin)):
    """System overview for the admin page. Protected by admin Basic-Auth."""
    store = get_vector_store()
    docs = store.list_documents()
    try:
        llm_health = get_llm().health()
    except Exception as exc:
        llm_health = {"provider": settings.resolved_llm_provider, "status": "error", "error": str(exc)}
    return {
        "documents": docs,
        "document_count": len(docs),
        "total_chunks": store.count(),
        "pending_uploads": len(_PENDING_UPLOADS),
        "embedding": {
            "provider": settings.EMBEDDING_PROVIDER,
            "model": settings.EMBEDDING_MODEL
            if settings.EMBEDDING_PROVIDER == "sentence_transformers"
            else settings.OPENAI_EMBEDDING_MODEL,
        },
        "llm": llm_health,
        "vector_store": store.health(),
        "retrieval": {
            "top_k": settings.RETRIEVAL_TOP_K,
            "min_relevance_score": settings.MIN_RELEVANCE_SCORE,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )
