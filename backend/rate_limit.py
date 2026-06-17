"""
Rate limiting via slowapi (a FastAPI-friendly wrapper around limits).

Why: protect the service (and your LLM bill) from abuse or runaway clients.
Limits are per client IP and configured through environment variables so you can
tune them per environment without code changes.

Wiring (done in main.py):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(SlowAPIMiddleware)

Then decorate routes with @limiter.limit(settings.RATE_LIMIT_CHAT), etc.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import settings

# `get_remote_address` keys limits on the client IP. Behind a proxy/load balancer
# you should configure the platform to forward the real IP (X-Forwarded-For).
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    headers_enabled=True,  # adds X-RateLimit-* response headers
)
