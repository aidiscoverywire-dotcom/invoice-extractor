"""End-to-end pipeline smoke test with a STUBBED OCR engine.

Covers: image decode -> OpenCV preprocess -> parse -> arithmetic validation ->
confidence gating -> SQLite persist -> review queue -> correction flow.
The only stubbed part is the PaddleOCR model itself (heavy, needs download);
everything around it is real.
"""
import io
import os

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

os.environ["DATABASE_URL"] = "sqlite:////tmp/smoke_test.db"
if os.path.exists("/tmp/smoke_test.db"):
    os.remove("/tmp/smoke_test.db")

from app.db.database import SessionLocal, init_db  # noqa: E402
from app.models.schemas import OcrLine  # noqa: E402
from app.pipeline import process_file  # noqa: E402
from app.review import queue as Q  # noqa: E402


def _line(text, conf, x0, y0, x1, y1, page=1):
    return OcrLine(text=text, confidence=conf,
                   bbox={"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
                   page=page)


# Fake OCR output shaped like the synthetic sample invoice.
FAKE_LINES = [
    _line("ACME TRADERS", 0.98, 80, 60, 600, 130),
    _line("42 Market Road, Lahore", 0.95, 80, 140, 500, 170),
    _line("Invoice No: INV-2026-0481", 0.99, 80, 300, 600, 340),
    _line("Invoice Date: 15 Sep 2026", 0.97, 80, 350, 600, 390),
    _line("Currency: PKR", 0.96, 80, 400, 400, 440),
    _line("Description", 0.99, 80, 520, 300, 550),
    _line("Qty", 0.99, 640, 520, 700, 550),
    _line("Unit Price", 0.99, 780, 520, 950, 550),
    _line("Amount", 0.99, 1000, 520, 1120, 550),
    _line("Widget A - premium grade", 0.94, 80, 590, 500, 620),
    _line("2", 0.93, 640, 590, 680, 620),
    _line("500.00", 0.95, 780, 590, 900, 620),
    _line("1000.00", 0.95, 1000, 590, 1120, 620),
    _line("Widget B - standard", 0.94, 80, 650, 500, 680),
    _line("5", 0.93, 640, 650, 680, 680),
    _line("120.00", 0.95, 780, 650, 900, 680),
    _line("600.00", 0.95, 1000, 650, 1120, 680),
    _line("Subtotal", 0.97, 780, 770, 950, 800),
    _line("1750.00", 0.96, 1000, 770, 1120, 800),
    _line("Tax (16%):", 0.90, 780, 825, 950, 855),
    _line("280.00", 0.96, 1000, 825, 1120, 855),
    _line("Grand Total:", 0.97, 780, 880, 980, 915),
    _line("2030.00", 0.96, 1000, 880, 1120, 915),
]


class StubEngine:
    def ocr_image(self, image, page=1):
        assert image is not None and image.size > 0  # real preprocessed pixels in
        return FAKE_LINES


def _sample_png() -> bytes:
    import sys
    sys.path.insert(0, "scripts")
    from make_sample_invoice import make_invoice
    make_invoice("/tmp/sample_inv.png")
    return open("/tmp/sample_inv.png", "rb").read()


def test_full_pipeline_smoke(monkeypatch):
    init_db()
    import app.pipeline as P
    monkeypatch.setattr(P, "get_engine", lambda: StubEngine())

    session = SessionLocal()
    result = process_file(_sample_png(), "invoice_001.png", session)

    assert result.status == "auto", result.notes
    assert result.fields.vendor.value == "ACME TRADERS"
    assert result.fields.invoice_number.value == "INV-2026-0481"
    assert result.fields.invoice_date.value == "2026-09-15"
    assert result.fields.currency.value == "PKR"
    assert result.fields.subtotal.value == 1750.00
    assert result.fields.tax.value == 280.00
    assert result.fields.total.value == 2030.00
    assert result.fields.arithmetic_ok is True
    assert len(result.line_items) == 2
    assert result.line_items[0].description == "Widget A - premium grade"
    assert result.line_items[0].qty == 2
    assert result.line_items[0].amount == 1000.00

    # Cache hit: same bytes -> same document, no reprocess
    again = process_file(_sample_png(), "invoice_001.png", session)
    assert again.document_id == result.document_id
    session.close()


def test_correction_propagates_to_document(monkeypatch):
    """Regression (verified 2026-09-29): a human correction must reach
    Document.fields_json, flip the status, and be visible on cache hits —
    the review queue must actually change something."""
    init_db()
    import app.pipeline as P

    weak_lines = [l for l in FAKE_LINES if l.text != "2030.00"]
    weak_lines.append(_line("2030.00", 0.60, 1000, 880, 1120, 915))  # weak total

    class WeakEngine:
        def ocr_image(self, image, page=1):
            return weak_lines

    monkeypatch.setattr(P, "get_engine", lambda: WeakEngine())
    session = SessionLocal()
    # force=True: same PNG bytes as the smoke test above would cache-hit,
    # and it exercises the reprocess path too.
    result = process_file(_sample_png(), "weak_total.png", session, force=True)
    assert result.status == "needs_review"
    assert "total" in result.low_confidence_fields

    items = Q.pending_items(session, document_id=result.document_id)
    item = next(i for i in items if i.field_name == "total")
    assert item.bbox_json is not None  # bbox came from the extractor, not text search

    Q.apply_correction(session, item.id, "2030.00", note="verified against scan")

    from app.db import models
    doc = session.get(models.Document, result.document_id)
    assert doc.fields_json["total"]["value"] == "2030.00"
    assert doc.fields_json["total"]["confidence"] == 1.0
    assert "human:corrected" in doc.fields_json["total"]["source"]
    assert doc.status == "verified"

    # Cache hit now returns the CORRECTED value, not the stale one.
    again = process_file(_sample_png(), "weak_total.png", session)
    assert again.fields.total.value == "2030.00"
    assert again.status == "verified"
    session.close()


def test_arithmetic_mismatch_forces_review(monkeypatch):
    """1750 + 280 != 2031 must never auto-accept, even at high confidence."""
    init_db()
    import app.pipeline as P

    bad_lines = [l for l in FAKE_LINES if l.text != "2030.00"]
    bad_lines.append(_line("2031.00", 0.99, 1000, 880, 1120, 915))

    class BadEngine:
        def ocr_image(self, image, page=1):
            return bad_lines

    monkeypatch.setattr(P, "get_engine", lambda: BadEngine())
    session = SessionLocal()
    result = process_file(_sample_png(), "bad_math.png", session, force=True)
    assert result.status == "needs_review"
    assert result.fields.arithmetic_ok is False
    assert any("arithmetic mismatch" in n for n in result.notes)
    session.close()


def test_review_queue_and_correction():
    init_db()
    session = SessionLocal()
    from app.db import models
    doc = models.Document(filename="t.png", sha256="abc123", pages=1,
                          status="needs_review", avg_confidence=0.8,
                          fields_json={}, line_items_json=[],
                          ocr_lines_json=[], notes_json=[])
    session.add(doc)
    session.commit()

    Q.add_review_items(session, doc.id, ["total"],
                       {"total": {"value": "2030.00", "confidence": 0.6,
                                  "source": "regex:label_ranked",
                                  "bbox": {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}}})
    pending = Q.pending_items(session, document_id=doc.id)
    assert len(pending) == 1 and pending[0].field_name == "total"
    assert pending[0].bbox_json == {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}

    fixed = Q.apply_correction(session, pending[0].id, "2050.00", note="verified vs scan")
    assert fixed.status == "resolved" and fixed.current_value == "2050.00"
    # Correction propagates to the stored document (not just the review item).
    doc2 = session.get(models.Document, doc.id)
    assert doc2.fields_json["total"]["value"] == "2050.00"
    assert doc2.fields_json["total"]["confidence"] == 1.0
    assert doc2.status == "verified"  # nothing left pending
    # history is append-only: old value preserved, never overwritten
    hist = session.query(models.Correction).filter_by(review_item_id=fixed.id).all()
    assert len(hist) == 1
    assert hist[0].old_value == "2030.00"
    assert hist[0].new_value == "2050.00"
    session.close()
