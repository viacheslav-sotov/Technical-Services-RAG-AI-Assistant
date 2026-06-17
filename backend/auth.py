"""
HTTP Basic authentication for admin / destructive routes.

We compare credentials using `secrets.compare_digest` to avoid timing attacks.
Credentials come from environment variables (ADMIN_USERNAME / ADMIN_PASSWORD),
never from source code.

Usage in a route:

    @app.get("/admin/stats")
    def stats(user: str = Depends(require_admin)):
        ...
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from backend.config import settings
from backend.utils import logger

_basic = HTTPBasic(auto_error=True)


def require_admin(
    credentials: HTTPBasicCredentials = Depends(_basic),
) -> str:
    """FastAPI dependency that enforces admin Basic-Auth credentials."""
    correct_user = secrets.compare_digest(
        credentials.username, settings.ADMIN_USERNAME
    )
    correct_pass = secrets.compare_digest(
        credentials.password, settings.ADMIN_PASSWORD
    )
    if not (correct_user and correct_pass):
        logger.warning("Rejected admin login for user '%s'", credentials.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
