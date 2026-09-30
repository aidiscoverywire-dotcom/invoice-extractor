"""Arithmetic validation + confidence gating (stdlib only).

Two-tier gating (research best practice):
  page avg >= AUTO        -> "auto"
  REVIEW <= page avg < AUTO, or any critical field weak -> "needs_review"
  page avg < REVIEW        -> "quarantined"

Critical fields: total, invoice_date. Arithmetic agreement
(subtotal + tax == total) is reported; a mismatch with all three values
present is escalated to needs_review by the pipeline (never auto-accept a
document whose own numbers disagree).

queue_fields (the review queue) covers critical fields below the floor PLUS
any other extracted field below the review threshold — not just the two
critical fields.
"""
from __future__ import annotations

CRITICAL_FIELDS = ("total", "invoice_date")
_TOLERANCE = 0.011  # one cent + float dust


def _num(v):
    """Coerce reviewer-entered strings ('2030.00') to float; None if impossible."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_arithmetic(subtotal: float | None, tax: float | None,
                     total: float | None) -> bool:
    subtotal = _num(subtotal)
    total = _num(total)
    if subtotal is None or total is None:
        return False
    tax = _num(tax) or 0.0
    return abs(subtotal + tax - total) <= _TOLERANCE


def gate(avg_confidence: float, fields: dict,
         auto_threshold: float = 0.92,
         review_threshold: float = 0.75,
         field_floor: float = 0.80) -> tuple[str, list[str], list[str], list[str]]:
    """Return (status, low_critical_fields, queue_fields, notes).

    queue_fields: every field that deserves a human look — critical fields
    below field_floor, plus non-critical fields below review_threshold.
    """
    low: list[str] = []
    queue: list[str] = []
    notes: list[str] = []

    for name in CRITICAL_FIELDS:
        f = fields.get(name, {})
        if not f.get("value") or f.get("confidence", 0.0) < field_floor:
            low.append(name)
            queue.append(name)

    for name, f in fields.items():
        if not isinstance(f, dict) or name in CRITICAL_FIELDS or name.startswith("_"):
            continue
        if f.get("value") and f.get("confidence", 0.0) < review_threshold:
            if name not in queue:
                queue.append(name)
            notes.append(f"low confidence on '{name}': {f['confidence']:.2f}")

    if avg_confidence < review_threshold:
        notes.append(f"page confidence {avg_confidence:.2f} below quarantine floor")
        return "quarantined", low, queue, notes
    if avg_confidence < auto_threshold or low:
        if low:
            notes.append("critical field(s) below floor: " + ", ".join(low))
        return "needs_review", low, queue, notes
    return "auto", low, queue, notes
