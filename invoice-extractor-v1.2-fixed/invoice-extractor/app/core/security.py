"""Optional API-key auth for /api/v1/*.

Set the API_KEY env var to require an `X-API-Key` header on every API route
(/health stays open). Unset (default) = open mode for local dev; the app logs
a warning at boot in that case. Invoices are PII — set this in production.

Key comparison is constant-time (hmac.compare_digest) and auth'd routes are
rate-limited per client (sliding 60s window, RATE_LIMIT_PER_MINUTE, 0 =
disabled). The limiter is in-memory per process — a safety net against
credential brute-forcing, not a replacement for a real WAF/edge limiter on
multi-worker deploys.
"""
from __future__ import annotations

import hmac
import time
from collections import defaultdict

from fastapi import Header, HTTPException, Request

from app.core.config import settings

_hits: dict[str, list[float]] = defaultdict(list)


def _safe_compare(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


def _client_id(request: Request, key: str | None) -> str:
    if key:
        return "key:" + key[-8:]
    host = request.client.host if request.client else "unknown"
    return "ip:" + host


def _check_rate_limit(request: Request, key: str | None) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    bucket = _hits[_client_id(request, key)]
    cutoff = now - 60.0
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        raise HTTPException(status_code=429,
                            detail="Rate limit exceeded — slow down and retry")
    bucket.append(now)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.api_key:
        return  # open mode (local dev)
    _check_rate_limit(request, x_api_key)
    if not x_api_key or not _safe_compare(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
