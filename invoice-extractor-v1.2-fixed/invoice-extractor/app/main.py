"""FastAPI application entrypoint.

Lifespan: create tables + warm up the OCR engine at boot so the first real
request doesn't pay the 8–12 s cold start.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.db.database import init_db

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Database ready (%s)", settings.database_url.split("://")[0])
    try:
        from app.ocr.engine import get_engine
        get_engine().warmup()  # absorbs model load + cold start before traffic
        app.state.ocr_ready = True
    except Exception as exc:  # noqa: BLE001 — model download/network can fail many ways
        # paddleocr missing or model download failed (e.g. no network) —
        # API still serves, /extract returns 503 with a clear message.
        log.warning("OCR engine unavailable at boot: %s", exc)
        app.state.ocr_ready = False
    if not settings.api_key:
        log.warning("API_KEY is not set — /api/v1/* is OPEN. Set API_KEY in production.")
    yield


app = FastAPI(
    title="Invoice Extractor",
    description="Zero-cost, open-source invoice/receipt data extraction (PaddleOCR + rules).",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "ocr_ready": getattr(app.state, "ocr_ready", False)}
