"""Unit tests for arithmetic validation + confidence gating (stdlib only)."""
from app.parser.validate import check_arithmetic, gate


def F(**kw):
    base = {"value": None, "confidence": 0.0, "source": ""}
    base.update(kw)
    return base


def fields(total_conf=0.95, date_conf=0.93):
    return {
        "total": F(value=2030.00, confidence=total_conf),
        "invoice_date": F(value="2026-09-15", confidence=date_conf),
        "vendor": F(value="ACME", confidence=0.97),
    }


def test_arithmetic_catches_single_digit_error():
    assert check_arithmetic(1750.00, 280.00, 2030.00) is True
    assert check_arithmetic(1750.00, 280.00, 2031.00) is False  # one digit off
    assert check_arithmetic(None, 280.00, 2030.00) is False


def test_gate_auto():
    status, low, queue, notes = gate(0.96, fields())
    assert status == "auto" and low == [] and queue == []


def test_gate_review_on_weak_page():
    status, low, queue, notes = gate(0.80, fields())
    assert status == "needs_review"


def test_gate_review_on_weak_critical_field():
    status, low, queue, notes = gate(0.96, fields(total_conf=0.62))
    assert status == "needs_review"
    assert "total" in low and "total" in queue


def test_gate_quarantine():
    status, low, queue, notes = gate(0.50, fields())
    assert status == "quarantined"


def test_gate_queues_weak_noncritical_field():
    f = fields()
    f["vendor"] = F(value="ACME", confidence=0.70)  # below review threshold
    status, low, queue, notes = gate(0.96, f)
    assert "vendor" in queue          # queued for review…
    assert status == "auto"           # …but doesn't block auto-accept
    assert any("vendor" in n for n in notes)
