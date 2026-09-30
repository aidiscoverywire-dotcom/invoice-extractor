"""Unit tests for the rule-based field parser (stdlib only)."""
from app.parser.fields import (
    extract_fields,
    find_currency,
    find_invoice_date,
    find_invoice_number,
    find_totals,
    find_vendor,
    parse_amount,
    repair_numeric,
)


def L(text, conf=0.95, bbox=None, page=1):
    return {"text": text, "confidence": conf, "bbox": bbox, "page": page}


def B(x0, y0, x1, y1):
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def test_parse_amount_formats():
    assert parse_amount("Rs 1,250.00") == 1250.00
    assert parse_amount("$1,250.00") == 1250.00
    assert parse_amount("2030.00") == 2030.00
    assert parse_amount("no digits here") is None


def test_repair_numeric_confusions():
    # OCR often reads 0 as O and 1 as l inside amounts.
    assert parse_amount("1,O00.00") == 1000.00
    assert repair_numeric("Invoice") == "Invoice"  # alpha text untouched


def test_invoice_number():
    lines = [L("ACME TRADERS"), L("Invoice No: INV-2026-0481")]
    f = find_invoice_number(lines)
    assert f["value"] == "INV-2026-0481"
    assert f["confidence"] == 0.95


def test_dates_normalize_to_iso():
    assert find_invoice_date([L("Date: 15/09/2026")])["value"] == "2026-09-15"
    assert find_invoice_date([L("Invoice Date: 15 Sep 2026")])["value"] == "2026-09-15"
    assert find_invoice_date([L("Dated: Sep 15, 2026")])["value"] == "2026-09-15"
    assert find_invoice_date([L("2026-09-15")])["value"] == "2026-09-15"


def test_total_uses_label_ranking_not_largest_amount():
    # The largest amount on the page is NOT the total — label must win.
    lines = [
        L("Widget A  5000.00"),          # decoy: bigger than the total
        L("Subtotal  1750.00"),
        L("Grand Total  2030.00"),
    ]
    t = find_totals(lines)
    assert t["total"]["value"] == 2030.00
    assert t["subtotal"]["value"] == 1750.00


def test_vendor_from_top_largest_region():
    lines = [
        L("ACME TRADERS", bbox=B(80, 60, 600, 130)),
        L("42 Market Road", bbox=B(80, 140, 400, 170)),
        L("Invoice No: INV-1", bbox=B(80, 900, 400, 930)),
    ]
    assert find_vendor(lines)["value"] == "ACME TRADERS"


def test_tax_percentage_not_mistaken_for_amount():
    # "Tax (16%):" must not yield 16.0 — the amount is on the next line.
    lines = [L("Tax (16%):"), L("280.00")]
    assert find_totals(lines)["tax"]["value"] == 280.00


def test_full_extraction_sample_invoice():
    lines = [
        L("ACME TRADERS", bbox=B(80, 60, 600, 130)),
        L("Invoice No: INV-2026-0481"),
        L("Invoice Date: 15 Sep 2026"),
        L("Currency: PKR"),
        L("Subtotal 1750.00"),
        L("Tax 280.00"),
        L("Grand Total 2030.00"),
    ]
    f = extract_fields(lines)
    assert f["vendor"]["value"] == "ACME TRADERS"
    assert f["invoice_number"]["value"] == "INV-2026-0481"
    assert f["invoice_date"]["value"] == "2026-09-15"
    assert f["currency"]["value"] == "PKR"
    assert f["total"]["value"] == 2030.00


# --- Regression tests for the verified parser bugs (2026-09-29) -------------

def test_invoice_number_not_greedy():
    # "Invoice Date:" must NOT yield invoice_number="Date".
    assert find_invoice_number([L("Invoice Date: 15 Sep 2026")])["value"] is None
    # "Invoice Total:" must NOT yield invoice_number="Total".
    assert find_invoice_number([L("Invoice Total: 2030.00")])["value"] is None
    # Real formats still work.
    assert find_invoice_number([L("Invoice No: INV-2026-0481")])["value"] == "INV-2026-0481"
    assert find_invoice_number([L("Invoice: AB-123")])["value"] == "AB-123"
    assert find_invoice_number([L("inv-2026-0481")])["value"] == "INV-2026-0481"


def test_comma_amount_without_decimals():
    # "Rs 1,250" was split into "1" -> 1.0.
    assert parse_amount("Rs 1,250") == 1250.0
    assert parse_amount("1,250") == 1250.0
    assert parse_amount("Rs 1,250.00") == 1250.00


def test_total_label_priority_across_lines():
    # "Net Total" on an earlier line must NOT beat "Grand Total".
    lines = [L("Net Total 1900.00"), L("Grand Total 2030.00")]
    assert find_totals(lines)["total"]["value"] == 2030.00


def test_total_qty_is_not_the_total():
    assert find_totals([L("Total Qty: 7")])["total"]["value"] is None


def test_header_row_does_not_feed_total():
    # A bare "Total" column header must not donate the first item's amount.
    lines = [L("Description Qty Unit Price Total"), L("Widget A 2 1000.00")]
    assert find_totals(lines)["total"]["value"] is None


def test_tax_excludes_invoice_metadata():
    assert find_totals([L("TAX INVOICE"), L("Invoice No: INV-2026-0481")])["tax"]["value"] is None
    assert find_totals([L("Tax ID: 1234567")])["tax"]["value"] is None
    # Real tax labels still work, including percentage-on-one-line.
    assert find_totals([L("Tax (16%):"), L("280.00")])["tax"]["value"] == 280.00
    assert find_totals([L("VAT 45.00")])["tax"]["value"] == 45.00


def test_currency_symbols_without_word_boundary():
    assert find_currency([L("Total: $100.00")])["value"] == "USD"
    assert find_currency([L("Total: €100.00")])["value"] == "EUR"
    assert find_currency([L("Currency: PKR")])["value"] == "PKR"


def test_date_excludes_due_and_updated():
    assert find_invoice_date([L("Last updated: 01/01/2020")])["value"] is None
    assert find_invoice_date([L("Due Date: 20/09/2026")])["value"] is None
    # ...but a real labeled date still works, incl. 2-digit year.
    assert find_invoice_date([L("Invoice Date: 15 Sep 2026")])["value"] == "2026-09-15"
    assert find_invoice_date([L("Date: 15/09/26")])["value"] == "2026-09-15"


def test_vendor_stays_on_first_page():
    lines = [L("PAGE TWO VENDOR", page=2, bbox=B(80, 60, 400, 100)),
             L("ACME Supplies Ltd", page=1, bbox=B(80, 60, 400, 100))]
    assert find_vendor(lines)["value"] == "ACME Supplies Ltd"


def test_repair_numeric_keeps_currency_prefix():
    assert repair_numeric("Rs 500") == "Rs 500"
    assert repair_numeric("1,O00") == "1,000"
    assert repair_numeric("Invoice") == "Invoice"
