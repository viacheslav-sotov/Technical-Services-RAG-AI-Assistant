"""
Shared utilities: logging setup, ID helpers, and small file helpers.

Kept dependency-free and side-effect-light so it can be imported anywhere.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path

from backend.config import settings

# A single set of allowed document types, reused by the API and the ingester.
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".markdown"}

_LOGGER_CONFIGURED = False


def setup_logging() -> logging.Logger:
    """
    Configure root logging once, with a consistent, greppable format.

    Logging (instead of print) is a production requirement: it gives us levels,
    timestamps, and the ability to route output to files/stdout for platforms
    like Render/Railway/Fly which capture stdout.
    """
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("tsrag")
    if _LOGGER_CONFIGURED:
        return logger

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.setLevel(level)
    logger.handlers = [handler]
    logger.propagate = False
    _LOGGER_CONFIGURED = True
    return logger


logger = setup_logging()


def new_id(prefix: str = "") -> str:
    """Generate a short unique id, optionally prefixed (e.g. 'doc')."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


def file_sha256(data: bytes) -> str:
    """Content hash, used to detect duplicate uploads cheaply."""
    return hashlib.sha256(data).hexdigest()


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if missing; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_extension(filename: str) -> str:
    """Return the lowercased file extension, including the leading dot."""
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    return safe_extension(filename) in ALLOWED_EXTENSIONS
