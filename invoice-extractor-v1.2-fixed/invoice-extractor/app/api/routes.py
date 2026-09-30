"""HTTP API: extract, documents, review queue, corrections.

- /extract is a plain `def` (NOT async): OCR is blocking, and FastAPI runs
  sync routes in a threadpool, so the event loop stays responsive for /health
  and friends while an extraction runs. Inference itself is serialized by a
  lock inside the OCR engine (PaddlePaddle is not thread-safe).
- Uploads are read in 1 MB chunks with an early abort: the size cap is
  enforced BEFORE the whole file sits in memory.
- All /api/v1/* routes require X-API-Key when the API_KEY env var is set.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import require_api_key
from app.db import models
from app.db.database import get_session
from app.models.schemas import (
    CorrectionIn, DocumentSummary, ExtractionResult, ReviewItemOut,
)
from app.pipeline import process_file
from app.review import queue as Q

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])

ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}
_ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")


def _safe_filename(name: str | None) -> str:
    """Basename + truncate: a >255-char filename used to blow up the
    String(255) column with an unhandled 500."""
    name = (name or "").strip().replace("\\", "/").split("/")[-1].strip() or "upload"
    if len(name) > 200:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 10:
            name = stem[: 200 - len(ext) - 1] + "." + ext
        else:
            name = name[:200]
    return name


def _safe_float(v) -> float | None:
    """Defensive float coercion: a junk value stored in fields_json (e.g. from
    before correction validation existed) must degrade to None, never 500."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_upload_limited(upload: UploadFile, limit_bytes: int) -> bytes:
    """Read the upload in chunks; abort as soon as the cap is exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise HTTPException(413, f"File exceeds {settings.max_file_mb} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/extract", response_model=ExtractionResult, summary="Extract invoice data from a file")
def extract(file: UploadFile = File(...),
            force: Annotated[bool, Query(description="Reprocess even if this file was extracted before")] = False,
            session: Session = Depends(get_session)):
    kind = ALLOWED_TYPES.get(file.content_type or "")
    if not kind and not (file.filename or "").lower().endswith(_ALLOWED_EXTS):
        raise HTTPException(400, "Only PDF, PNG, JPG, WEBP files are accepted")
    data = _read_upload_limited(file, settings.max_file_mb * 1024 * 1024)
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        return process_file(data, _safe_filename(file.filename), session, force=force)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:  # e.g. paddleocr not installed
        raise HTTPException(503, str(exc)) from exc


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(
        status: Annotated[str | None, Query(pattern="^(auto|needs_review|quarantined|verified)$")] = None,
        limit: Annotated[int, Query(le=200)] = 50,
        session: Session = Depends(get_session)):
    q = session.query(models.Document).order_by(models.Document.created_at.desc())
    if status:
        q = q.filter_by(status=status)
    docs = q.limit(limit).all()
    return [DocumentSummary(
        id=d.id, filename=d.filename, pages=d.pages, status=d.status,
        avg_confidence=d.avg_confidence,
        total=_safe_float((d.fields_json or {}).get("total", {}).get("value")),
        created_at=d.created_at) for d in docs]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str,
                 include_ocr_lines: Annotated[bool, Query(description="Set false to skip the OCR lines payload")] = True,
                 ocr_lines_limit: Annotated[int, Query(ge=0, le=50000, description="Max OCR lines returned")] = 2000,
                 ocr_lines_offset: Annotated[int, Query(ge=0, description="Skip this many OCR lines first")] = 0,
                 session: Session = Depends(get_session)):
    doc = session.get(models.Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    lines = doc.ocr_lines_json or []
    if include_ocr_lines:
        page = lines[ocr_lines_offset: ocr_lines_offset + ocr_lines_limit]
    else:
        page = []
    return {
        "id": doc.id, "filename": doc.filename, "pages": doc.pages,
        "status": doc.status, "avg_confidence": doc.avg_confidence,
        "fields": doc.fields_json, "line_items": doc.line_items_json,
        "ocr_lines": page, "ocr_lines_total": len(lines),
        "notes": doc.notes_json,
        "created_at": doc.created_at,
    }


@router.get("/review/pending", response_model=list[ReviewItemOut])
def review_pending(limit: Annotated[int, Query(le=500)] = 100,
                   session: Session = Depends(get_session)):
    return [ReviewItemOut(
        id=i.id, document_id=i.document_id, field_name=i.field_name,
        current_value=i.current_value, reason=i.reason, bbox=i.bbox_json,
        status=i.status, created_at=i.created_at)
        for i in Q.pending_items(session, limit)]


@router.post("/review/{item_id}/correct", summary="Submit a human correction (logged, append-only, propagates to the document)")
def correct(item_id: int, body: CorrectionIn, session: Session = Depends(get_session)):
    try:
        item = Q.apply_correction(session, item_id, body.corrected_value, body.note)
    except KeyError:
        raise HTTPException(404, "Review item not found")
    except ValueError as exc:  # correction failed type validation
        raise HTTPException(422, str(exc)) from exc
    return {"item_id": item.id, "status": item.status,
            "corrected_value": item.current_value}
