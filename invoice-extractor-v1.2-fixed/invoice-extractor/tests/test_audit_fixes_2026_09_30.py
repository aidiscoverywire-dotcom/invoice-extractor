"""Regression tests for the 2026-09-29 code audit (verified 2026-09-30).

Each test reproduces one audit finding against the fixed code. Numbering
matches the audit PDF: #1-5 critical, #6-7 eval, #8-16 minor/hardening.
"""
import json
import sys

import pytest

sys.path.insert(0, "eval")

from app.api import routes
from app.core import security
from app.core.config import settings
from app.db import models
from app.db.models import Base
from app.models import schemas
from app.models.schemas import CorrectionIn, DocumentSummary
from app.parser import tables
from app.parser.fields import find_currency, find_invoice_date, parse_amount
from app.parser.validate import CRITICAL_FIELDS
from app.review import queue as Q
from evaluate import _norm, field_scores, main as eval_main

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def L(text, conf=0.95, bbox=None, page=1):
    return {"text": text, "confidence": conf, "bbox": bbox, "page": page}


def LB(text, x0, y0, x1, y1, conf=0.95, page=1):
    return {"text": text, "confidence": conf,
            "bbox": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1},
            "page": page}


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, autoflush=False)()
    yield s
    s.close()


_DOC_SEQ = 0


def _doc_with_item(s, status="needs_review", value=2030.0, field="total"):
    global _DOC_SEQ
    _DOC_SEQ += 1
    did = f"d{_DOC_SEQ}"
    doc = models.Document(id=did, filename="a.pdf", sha256=f"{_DOC_SEQ:064d}",
                          pages=1, avg_confidence=0.8, status=status,
                          fields_json={field: {"value": value, "confidence": 0.5,
                                               "source": "regex:label_ranked"}})
    s.add(doc)
    s.commit()
    item = models.ReviewItem(document_id=did, field_name=field,
                             current_value=str(value), reason="weak",
                             status="pending")
    s.add(item)
    s.commit()
    return doc, item


# ---------------------------------------------------------------- #1: crash
def test_correction_rejects_non_numeric_for_money_fields(session):
    _, item = _doc_with_item(session)
    with pytest.raises(ValueError, match="must be a number"):
        Q.apply_correction(session, item.id, "N/A")
    # nothing was written: item still pending, no correction row
    s_item = session.get(models.ReviewItem, item.id)
    assert s_item.status == "pending"
    assert session.query(models.Correction).count() == 0


def test_correction_rejects_bad_date(session):
    _, item = _doc_with_item(session, field="invoice_date", value="2026-09-15")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        Q.apply_correction(session, item.id, "not a date")


def test_list_documents_survives_junk_total(session):
    # Pre-fix data (or any junk) in fields_json must degrade to None, never 500.
    doc, _ = _doc_with_item(session, value="N/A")
    out = routes.list_documents(status=None, limit=50, session=session)
    assert out[0].total is None
    # and a numeric string still coerces
    doc.fields_json = {"total": {"value": "2030.00"}}
    session.commit()
    out = routes.list_documents(status=None, limit=50, session=session)
    assert out[0].total == 2030.00


# ------------------------------------------------------- #2: negative signs
def test_negative_amounts_keep_sign():
    assert parse_amount("(500.00)") == -500.0
    assert parse_amount("-500.00") == -500.0
    assert parse_amount("Rs -500.00") == -500.0
    assert parse_amount("($1,250.00)") == -1250.0
    assert parse_amount("(2.030,00)") == -2030.0


def test_hyphen_in_token_is_not_a_negative():
    assert parse_amount("ref-500") == 500.0
    assert parse_amount("2026-09-15") == 2026.0
    assert parse_amount("Rs 500.00") == 500.0  # positive still positive


# ------------------------------------------------------- #3: EU number format
def test_european_number_format():
    assert parse_amount("2.030,00") == 2030.00
    assert parse_amount("1.234.567,89") == 1234567.89
    assert parse_amount("12,50") == 12.5
    assert parse_amount("1.234.567") == 1234567.0


def test_us_number_format_unchanged():
    assert parse_amount("1,250.00") == 1250.00
    assert parse_amount("1,250,000") == 1250000.0
    assert parse_amount("$1,250.00") == 1250.00
    assert parse_amount("2030.00") == 2030.00


# ------------------------------------------------------- #4: quarantined flip
def test_quarantined_doc_becomes_verified(session):
    doc, item = _doc_with_item(session, status="quarantined")
    Q.apply_correction(session, item.id, "2030.00")
    session.refresh(doc)
    assert doc.status == "verified"


# ------------------------------------------------------- #5: ambiguous dates
def test_ambiguous_date_carries_warning():
    r = find_invoice_date([L("Invoice Date: 05/06/2026")])
    assert r["value"] == "2026-06-05"
    assert r["note"] and "ambiguous" in r["note"]


def test_ambiguous_date_month_first_mode():
    r = find_invoice_date([L("Invoice Date: 05/06/2026")], day_first=False)
    assert r["value"] == "2026-05-06"
    assert r["note"] and "month-first" in r["note"]


def test_unambiguous_date_has_no_warning():
    r = find_invoice_date([L("Invoice Date: 15/09/2026")])
    assert r["value"] == "2026-09-15" and r["note"] is None
    r = find_invoice_date([L("Invoice Date: 06/15/2026")])  # US fallback
    assert r["value"] == "2026-06-15" and r["note"] is None


# ------------------------------------------------------- #6/#7: eval
def test_eval_norm_numeric_type_mismatch():
    assert _norm(2030.0) == _norm("2030.00") == _norm(2030)
    s = field_scores({"total": "2030.00"}, {"total": 2030.0}, ["total"])
    assert (s["precision"], s["recall"], s["f1"]) == (1.0, 1.0, 1.0)


def test_eval_aggregate_printed(tmp_path, capsys, monkeypatch):
    gt = {"invoices": [
        {"id": "a", "fields": {"total": 2030.0, "vendor": "Acme"}},
        {"id": "b", "fields": {"total": 100.0, "vendor": "Beta"}},
    ]}
    pred = {"a": {"fields": {"total": "2030.00", "vendor": "Acme"}},
            "b": {"fields": {"total": 999.0, "vendor": "Beta"}}}
    p, g = tmp_path / "p.json", tmp_path / "g.json"
    p.write_text(json.dumps(pred))
    g.write_text(json.dumps(gt))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--pred", str(p), "--gt", str(g)])
    eval_main()
    out = capsys.readouterr().out
    assert "Aggregate MICRO" in out and "Aggregate MACRO" in out
    assert "total F1=" in out  # per-field line


# ------------------------------------------------------- #8: auth hardening
def test_api_key_compare_is_constant_time():
    assert security._safe_compare("secret-key", "secret-key") is True
    assert security._safe_compare("secret-key", "secret-keY") is False
    assert security._safe_compare("", "secret-key") is False


def _req():
    from starlette.requests import Request
    return Request({"type": "http", "http_version": "1.1", "method": "GET",
                    "scheme": "http", "path": "/", "headers": [],
                    "client": ("127.0.0.1", 5000)})


def test_auth_flow_and_rate_limit(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    security._hits.clear()
    monkeypatch.setattr(settings, "api_key", "topsecret")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    # wrong key -> 401 (uses its own per-key bucket, doesn't burn the good one)
    with pytest.raises(HTTPException) as e:
        asyncio.run(security.require_api_key(_req(), "wrong"))
    assert e.value.status_code == 401
    # two good requests pass, the third inside the window -> 429
    asyncio.run(security.require_api_key(_req(), "topsecret"))
    asyncio.run(security.require_api_key(_req(), "topsecret"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(security.require_api_key(_req(), "topsecret"))
    assert e.value.status_code == 429
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 120)


# ------------------------------------------------------- #9: filename length
def test_safe_filename():
    long = "a" * 300 + ".pdf"
    out = routes._safe_filename(long)
    assert len(out) <= 200 and out.endswith(".pdf")
    assert routes._safe_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert routes._safe_filename(None) == "upload"
    assert routes._safe_filename("invoice.pdf") == "invoice.pdf"


# ------------------------------------------------------- #10: blank correction
def test_correction_rejects_whitespace_only():
    with pytest.raises(Exception):
        CorrectionIn(corrected_value="   ")


# ------------------------------------------------------- #11: reason threshold
def test_review_reason_states_real_threshold(session):
    doc, _ = _doc_with_item(session)
    Q.add_review_items(
        session, doc.id, ["total", "vendor"],
        {"total": {"value": 1, "confidence": 0.5, "source": "x"},
         "vendor": {"value": "Acme", "confidence": 0.5, "source": "y"}},
        thresholds={"total": 0.80, "vendor": 0.75})
    reasons = {i.field_name: i.reason for i in
               Q.pending_items(session, document_id=doc.id)
               if i.reason.startswith("Weak confidence")}
    assert "(floor 0.80)" in reasons["total"]
    assert "(floor 0.75)" in reasons["vendor"]  # not the misleading 0.80


# ------------------------------------------------------- #12: utcnow
def test_created_at_default_is_timezone_aware(session):
    # The deprecated datetime.utcnow() is gone: the column default now
    # produces a timezone-aware UTC instant in Python...
    assert models._utcnow().tzinfo is not None
    # ...and the stored wall-clock time is still UTC, as before.
    doc = models.Document(filename="t.pdf", sha256="1" * 64, pages=1)
    session.add(doc)
    session.commit()
    session.refresh(doc)
    from datetime import datetime, timezone
    delta = abs((datetime.now(timezone.utc).replace(tzinfo=None)
                 - doc.created_at).total_seconds())
    assert delta < 60


# ------------------------------------------------------- #13: delete tree, no SAWarning
@pytest.mark.filterwarnings("error::sqlalchemy.exc.SAWarning")
def test_delete_doc_tree_no_deprecated_subquery_in(session):
    from app.pipeline import _delete_doc_tree
    doc, item = _doc_with_item(session)
    Q.apply_correction(session, item.id, "2030.00")
    _delete_doc_tree(session, doc)
    assert session.get(models.Document, doc.id) is None
    assert session.query(models.ReviewItem).count() == 0
    assert session.query(models.Correction).count() == 0


# ------------------------------------------------------- #14: currency scan
def test_currency_found_deep_in_document():
    lines = [L(f"filler {i}") for i in range(60)]
    lines.append(L("amount due EUR 500"))
    assert find_currency(lines)["value"] == "EUR"


# ------------------------------------------------------- #15: single-char rows
def test_single_char_numeric_row_kept():
    header = [LB("Description", 80, 520, 300, 550), LB("Qty", 640, 520, 700, 550),
              LB("Unit Price", 780, 520, 950, 550), LB("Amount", 1000, 520, 1120, 550)]
    rows = header + [LB("5", 640, 590, 680, 620)]      # qty-only row
    rows += [LB("x", 80, 650, 100, 680)]               # noise: still dropped
    items = tables.extract_line_items(rows)
    assert len(items) == 1
    assert items[0]["qty"] == 5


# ------------------------------------------------------- #16: ocr_lines pagination
def test_get_document_ocr_lines_pagination(session):
    doc = models.Document(id="d9", filename="p.pdf", sha256="z" * 64, pages=2,
                          status="auto", avg_confidence=0.9,
                          ocr_lines_json=[{"text": f"l{i}"} for i in range(10)])
    session.add(doc)
    session.commit()
    full = routes.get_document("d9", session=session)
    assert full["ocr_lines_total"] == 10 and len(full["ocr_lines"]) == 10
    paged = routes.get_document("d9", ocr_lines_limit=3, ocr_lines_offset=4,
                                session=session)
    assert [l["text"] for l in paged["ocr_lines"]] == ["l4", "l5", "l6"]
    assert paged["ocr_lines_total"] == 10
    skipped = routes.get_document("d9", include_ocr_lines=False, session=session)
    assert skipped["ocr_lines"] == [] and skipped["ocr_lines_total"] == 10


# ------------------------------------------------------- misc: validation paths
def test_validate_correction_value_currency_and_vendor(session):
    _, item = _doc_with_item(session, field="currency", value="PKR")
    assert Q.validate_correction_value("currency", "usd") == "USD"
    with pytest.raises(ValueError):
        Q.validate_correction_value("currency", "USDD")
    # vendor accepts any non-blank text
    _, item2 = _doc_with_item(session, field="vendor", value="Acme")
    Q.apply_correction(session, item2.id, "  Beta Corp  ")
    assert session.get(models.ReviewItem, item2.id).current_value == "Beta Corp"
