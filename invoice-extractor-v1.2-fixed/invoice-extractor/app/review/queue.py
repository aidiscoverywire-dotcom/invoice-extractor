"""Review queue helpers: create items for uncertain fields, apply corrections.

The bbox on each review item comes from the extractor itself (the source
line's bbox recorded during parsing) — never from fragile text re-matching.
"""
from __future__ import annotations

import re
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import models
from app.parser.fields import parse_amount
from app.parser.validate import CRITICAL_FIELDS

# Corrections are type-checked per field: a non-numeric "N/A" in `total`
# used to be stored raw and then 500 the whole GET /documents endpoint when
# Pydantic tried to coerce it to float.
_NUMERIC_FIELDS = {"subtotal", "tax", "total"}
_DATE_FIELDS = {"invoice_date"}


def validate_correction_value(field_name: str, value: str) -> str:
    """Type-check a reviewer correction. Returns the stripped value.

    Raises ValueError (mapped to HTTP 422 by the API) when the value cannot
    possibly belong to the field — e.g. "N/A" for a money field.
    """
    v = value.strip()
    if field_name in _NUMERIC_FIELDS:
        if parse_amount(v) is None:
            raise ValueError(
                f"correction for '{field_name}' must be a number, got {value!r}")
    elif field_name in _DATE_FIELDS:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"correction for '{field_name}' must be a date YYYY-MM-DD, "
                f"got {value!r}") from None
    elif field_name == "currency":
        if not re.fullmatch(r"[A-Za-z]{3}", v):
            raise ValueError(
                f"correction for 'currency' must be a 3-letter code, "
                f"got {value!r}")
        v = v.upper()
    return v


def add_review_items(session: Session, document_id: str,
                     field_names: list[str], fields: dict,
                     thresholds: dict[str, float] | None = None) -> None:
    """One review item per uncertain field, carrying the human-readable reason
    and the bbox so a UI can highlight the exact region.

    thresholds maps field -> the floor that queued it (critical fields use
    field_confidence_floor, the rest review_threshold) so the reason text
    states the REAL threshold instead of a misleading constant.
    """
    default_floor = settings.field_confidence_floor
    for name in field_names:
        f = fields.get(name, {})
        if not isinstance(f, dict):
            continue
        floor = (thresholds or {}).get(name, default_floor)
        session.add(models.ReviewItem(
            document_id=document_id,
            field_name=name,
            current_value=str(f.get("value")) if f.get("value") is not None else None,
            reason=(f"Weak confidence {f.get('confidence', 0.0):.2f} "
                    f"(floor {floor:.2f}); source={f.get('source', 'unknown')}. "
                    f"Verify against the highlighted region."),
            bbox_json=f.get("bbox"),
        ))
    session.commit()


def apply_correction(session: Session, item_id: int, new_value: str,
                     note: str = "") -> models.ReviewItem:
    """Append-only correction that ALSO propagates to the stored Document.

    - The correction is logged (old value never lost).
    - Document.fields_json is updated: the corrected field gets the new value
      with confidence 1.0 and a human:corrected source marker. The JSON
      column is REASSIGNED (SQLAlchemy does not track in-place mutation).
    - A note is appended to the document.
    - When no pending review items remain and the document was in review,
      its status becomes "verified".
    """
    item = session.get(models.ReviewItem, item_id)
    if item is None:
        raise KeyError(f"review item {item_id} not found")
    # Validate BEFORE anything is written: a non-numeric "N/A" for `total`
    # used to poison fields_json and 500 the /documents endpoints.
    new_value = validate_correction_value(item.field_name, new_value)
    session.add(models.Correction(
        review_item_id=item.id,
        old_value=item.current_value,
        new_value=new_value,
        note=note,
    ))
    item.current_value = new_value
    item.status = "resolved"

    doc = session.get(models.Document, item.document_id)
    if doc is not None:
        fj = dict(doc.fields_json or {})
        entry = dict(fj.get(item.field_name, {}))
        source = (entry.get("source", "") + "+human:corrected").lstrip("+")
        entry.update({"value": new_value, "confidence": 1.0, "source": source})
        fj[item.field_name] = entry
        doc.fields_json = fj  # reassign: in-place JSON mutation isn't tracked
        notes = list(doc.notes_json or [])
        notes.append(f"reviewer corrected '{item.field_name}' -> {new_value}")
        doc.notes_json = notes
        session.flush()  # Session uses autoflush=False: make the "resolved"
        # status visible to the count query below.
        remaining = session.query(models.ReviewItem).filter_by(
            document_id=doc.id, status="pending").count()
        # Both review states clear to verified once nothing is pending —
        # quarantined documents used to be stuck forever.
        if remaining == 0 and doc.status in ("needs_review", "quarantined"):
            doc.status = "verified"

    session.commit()
    session.refresh(item)
    return item


def pending_items(session: Session, limit: int = 100,
                  document_id: str | None = None) -> list[models.ReviewItem]:
    q = (session.query(models.ReviewItem)
         .filter(models.ReviewItem.status == "pending"))
    if document_id is not None:
        q = q.filter(models.ReviewItem.document_id == document_id)
    return q.order_by(models.ReviewItem.created_at.desc()).limit(limit).all()
