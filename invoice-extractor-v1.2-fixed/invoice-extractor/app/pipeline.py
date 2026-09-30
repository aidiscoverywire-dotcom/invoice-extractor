"""End-to-end extraction pipeline.

upload bytes
  -> sha256 dedupe (content-hash cache; IntegrityError-safe against races;
     force=True deletes the old tree and reprocesses)
  -> PDF -> 300 DPI raster | image as-is
  -> preprocess (deskew/denoise; aggressive mode is opt-in)
  -> PaddleOCR per page (+ optional Tesseract fallback on weak regions)
  -> fields + line items
  -> arithmetic validation (a mismatch with all three values present
     forces needs_review — never auto-accept disagreeing numbers)
  -> confidence gating -> persist -> review queue
"""
from __future__ import annotations

import hashlib
import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import models
from app.models.schemas import ExtractionResult, FieldValue, InvoiceFields, LineItem
from app.ocr import preprocess as pp
from app.ocr.engine import get_engine
from app.parser import fields as F
from app.parser import tables as T
from app.parser import validate as V
from app.review import queue as Q

log = logging.getLogger(__name__)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rasterize(data: bytes, filename: str) -> list:
    if pp.is_pdf(data):
        images = pp.pdf_to_images(data, dpi=300, max_pages=settings.max_pages)
        if not images:
            raise ValueError("PDF contained no renderable pages")
        return images
    import cv2
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {filename}")
    return [img]


def _delete_doc_tree(session: Session, doc: models.Document) -> None:
    """Remove a document and its review trail (used by force reprocess)."""
    item_ids = select(models.ReviewItem.id).where(
        models.ReviewItem.document_id == doc.id)
    session.execute(delete(models.Correction).where(
        models.Correction.review_item_id.in_(item_ids)))
    session.execute(delete(models.ReviewItem).where(
        models.ReviewItem.document_id == doc.id))
    session.delete(doc)
    session.commit()


def process_file(data: bytes, filename: str, session: Session,
                 force: bool = False) -> ExtractionResult:
    digest = _sha256(data)
    existing = session.query(models.Document).filter_by(sha256=digest).first()
    if existing is not None and not force:
        log.info("Cache hit for %s (%s)", filename, digest[:8])
        return _to_result(existing, session)
    if existing is not None and force:
        log.info("Force reprocess for %s (%s)", filename, digest[:8])
        _delete_doc_tree(session, existing)

    images = _rasterize(data, filename)
    engine = get_engine()

    ocr_lines: list[dict] = []
    for page_no, img in enumerate(images, start=1):
        img = pp.ensure_min_dpi(img)
        clean = pp.preprocess(img, aggressive=settings.preprocess_aggressive)
        lines = engine.ocr_image(clean, page=page_no)
        log.info("Page %d: %d OCR lines", page_no, len(lines))

        if settings.enable_tesseract_fallback:
            from app.ocr import fallback as fb
            if fb.tesseract_available():
                avg = sum(l.confidence for l in lines) / max(len(lines), 1)
                if avg < settings.review_threshold:
                    log.info("Page %d weak (%.2f) — Tesseract on weak regions only",
                             page_no, avg)
                    lines = fb.merge_weak_regions(
                        lines, fb.tesseract_lines(clean, page=page_no))

        ocr_lines.extend(l.model_dump() for l in lines)

    if not ocr_lines:
        raise ValueError("OCR returned no text — blank or unreadable document")

    extracted = F.extract_fields(ocr_lines, day_first=settings.date_day_first)
    line_items = T.extract_line_items(ocr_lines)

    sub = extracted["subtotal"]["value"]
    tax = extracted["tax"]["value"]
    tot = extracted["total"]["value"]
    arithmetic_ok = V.check_arithmetic(sub, tax, tot)
    if arithmetic_ok:
        extracted["total"]["source"] += "+arithmetic:validated"
    extracted["_arithmetic_ok"] = arithmetic_ok

    avg_conf = round(sum(l["confidence"] for l in ocr_lines) / len(ocr_lines), 3)
    status, low_fields, queue_fields, notes = V.gate(
        avg_conf, extracted,
        auto_threshold=settings.auto_threshold,
        review_threshold=settings.review_threshold,
        field_floor=settings.field_confidence_floor,
    )
    # A document whose own numbers disagree is strong evidence of an OCR
    # error — never auto-accept it, even at high confidence.
    if not arithmetic_ok and all(v is not None for v in (sub, tax, tot)):
        status = "needs_review"
        notes.append(f"arithmetic mismatch: {sub} + {tax} != {tot} — flagged for review")
        for name in ("total", "subtotal", "tax"):
            if name not in queue_fields:
                queue_fields.append(name)

    # Ambiguous numeric dates are never silent: the parser flags them and the
    # note lands on the document for a human to verify.
    date_note = (extracted.get("invoice_date") or {}).get("note")
    if date_note:
        notes.append(date_note)

    doc = models.Document(
        filename=filename,
        sha256=digest,
        pages=len(images),
        avg_confidence=avg_conf,
        status=status,
        fields_json={k: v for k, v in extracted.items() if not k.startswith("_")},
        line_items_json=line_items,
        ocr_lines_json=ocr_lines,
        notes_json=notes,
    )
    session.add(doc)
    try:
        session.commit()
    except IntegrityError:
        # Dedupe race: a concurrent upload of the same file won the commit.
        session.rollback()
        existing = session.query(models.Document).filter_by(sha256=digest).first()
        log.info("Dedupe race on %s — returning the existing document", filename)
        return _to_result(existing, session)
    session.refresh(doc)

    if queue_fields:
        # Tell the review queue which floor queued each field so the reason
        # text states the real threshold (critical 0.80 vs the rest 0.75).
        thresholds = {
            n: (settings.field_confidence_floor
                if n in V.CRITICAL_FIELDS else settings.review_threshold)
            for n in queue_fields
        }
        Q.add_review_items(session, doc.id, queue_fields, extracted,
                           thresholds=thresholds)

    return _to_result(doc, session, queue_fields=queue_fields)


def _to_result(doc: models.Document, session: Session,
               queue_fields: list[str] | None = None) -> ExtractionResult:
    fq = doc.fields_json or {}

    def fv(name: str) -> FieldValue:
        f = fq.get(name, {})
        return FieldValue(value=f.get("value"), confidence=f.get("confidence", 0.0),
                          source=f.get("source", ""))

    sub_v = (fq.get("subtotal") or {}).get("value")
    tax_v = (fq.get("tax") or {}).get("value")
    tot_v = (fq.get("total") or {}).get("value")
    arith = (V.check_arithmetic(sub_v, tax_v, tot_v)
             if sub_v is not None and tot_v is not None else False)

    fields = InvoiceFields(
        vendor=fv("vendor"), invoice_number=fv("invoice_number"),
        invoice_date=fv("invoice_date"), currency=fv("currency"),
        subtotal=fv("subtotal"), tax=fv("tax"), total=fv("total"),
        arithmetic_ok=bool(arith),
    )
    items = [LineItem(**it) for it in (doc.line_items_json or [])]
    if queue_fields is None:
        queue_fields = [i.field_name for i in
                        Q.pending_items(session, document_id=doc.id)]
    return ExtractionResult(
        document_id=doc.id,
        filename=doc.filename,
        pages=doc.pages,
        fields=fields,
        line_items=items,
        avg_confidence=doc.avg_confidence,
        status=doc.status,
        low_confidence_fields=queue_fields,
        notes=doc.notes_json or [],
    )
