"""Geometric line-item extraction from OCR boxes (stdlib only).

Approach: cluster OCR lines into rows by vertical overlap, split rows into
cells by horizontal gaps, detect the header row by keywords, then read data
rows until the totals section. Works for clean invoices; for complex or
borderless tables the roadmap upgrade is PP-StructureV3 (SLANet table mode).

Input: list of {text, confidence, bbox, page} dicts. Lines WITHOUT bboxes are
ignored when boxed lines exist; otherwise each line becomes its own row
(degraded mode).
"""
from __future__ import annotations

from app.parser.fields import _conf, _text, _bbox, _page, _line_amounts, parse_amount

_HEADER_KEYWORDS = {"description", "item", "particulars", "qty", "quantity",
                    "qnt", "price", "rate", "unit", "amount", "total", "pcs"}
_TOTALS_HINTS = ("subtotal", "sub total", "tax", "vat", "gst", "grand total",
                 "amount due", "balance due")


def _center_y(b) -> float:
    return (b["y_min"] + b["y_max"]) / 2


def _center_x(b) -> float:
    return (b["x_min"] + b["x_max"]) / 2


def cluster_rows(lines: list) -> list[list[dict]]:
    """Group lines into visual rows by vertical overlap.

    Rows NEVER merge across pages: a page-2 line at the same y-coordinate as
    a page-1 line lands in a different row.
    """
    boxed = [l for l in lines if _bbox(l)]
    if not boxed:
        return [[l] for l in lines]
    by_page: dict[int, list] = {}
    for l in boxed:
        by_page.setdefault(_page(l), []).append(l)
    rows: list[list[dict]] = []
    for pg in sorted(by_page):
        rows.extend(_cluster_one_page(by_page[pg]))
    return rows


def _cluster_one_page(boxed: list) -> list[list[dict]]:
    heights = [b["y_max"] - b["y_min"] for b in (_bbox(l) for l in boxed)]
    med_h = sorted(heights)[len(heights) // 2] or 10.0
    ordered = sorted(boxed, key=lambda l: _center_y(_bbox(l)))
    rows: list[list[dict]] = []
    for line in ordered:
        cy = _center_y(_bbox(line))
        placed = False
        for row in rows:
            rcy = sum(_center_y(_bbox(l)) for l in row) / len(row)
            if abs(cy - rcy) < med_h * 0.6:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: _bbox(l)["x_min"])
    return rows


def _is_header(row: list[dict]) -> bool:
    words = set()
    for l in row:
        words.update(_text(l).lower().split())
    return len(words & _HEADER_KEYWORDS) >= 2


def _is_totals_row(row: list[dict]) -> bool:
    joined = " ".join(_text(l) for l in row).lower()
    return any(h in joined for h in _TOTALS_HINTS)


def _header_role(text: str) -> str | None:
    """Classify a header cell: description | qty | unit_price | amount."""
    t = text.strip().lower()
    if any(k in t for k in ("qty", "quantity", "qnt", "qnty", "pcs", "units")):
        return "qty"
    if "unit" in t and "price" in t:
        return "unit_price"
    if t in ("rate", "price", "unit price", "unit rate"):
        return "unit_price"
    if "amount" in t or t in ("total", "line total", "extended"):
        return "amount"
    if any(k in t for k in ("description", "item", "particular", "product", "service", "detail")):
        return "description"
    return None


def extract_line_items(lines: list) -> list[dict]:
    rows = cluster_rows(lines)
    header_idx = next((i for i, r in enumerate(rows) if _is_header(r)), None)
    if header_idx is None:
        return []

    # Map each header column's x-center to its role — data cells are assigned
    # to the nearest header column instead of guessing by position.
    header = rows[header_idx]
    columns: list[tuple[float, str | None]] = []
    for cell in header:
        b = _bbox(cell)
        cx = (_center_x(b)) if b else 0.0
        columns.append((cx, _header_role(_text(cell))))

    items: list[dict] = []
    for row in rows[header_idx + 1:]:
        if _is_totals_row(row):
            break
        texts = [_text(l).strip() for l in row]
        if not any(texts):
            continue
        # Single-character rows are usually OCR noise — keep them only when
        # the cell is numeric (a real qty/amount cell), e.g. a row of ["5"].
        if (sum(len(t) for t in texts) < 2
                and not any(parse_amount(t) is not None for t in texts)):
            continue

        item = {"description": "", "qty": None, "unit_price": None,
                "amount": None, "confidence": 0.0}
        desc_parts: list[str] = []
        for cell in row:
            b = _bbox(cell)
            cx = _center_x(b) if b else 0.0
            role = min(columns, key=lambda c: abs(c[0] - cx))[1] if columns else None
            text = _text(cell).strip()
            if role in ("description", None):
                # Unclassified columns are treated as description text.
                if role is None or parse_amount(text) is None:
                    desc_parts.append(text)
            elif role == "qty":
                q = parse_amount(text)
                if q is not None and 0 < q < 100000 and q == int(q):
                    item["qty"] = q
            elif role == "unit_price":
                item["unit_price"] = parse_amount(text)
            elif role == "amount":
                item["amount"] = parse_amount(text)

        # Fallback: no amount captured but the row has trailing numbers
        # (e.g. header roles missed) — last number is the amount.
        if item["amount"] is None:
            flat: list[float] = []
            for t in texts:
                flat.extend(_line_amounts(t))
            if flat:
                item["amount"] = flat[-1]

        item["description"] = " ".join(desc_parts).strip(" -|") or " ".join(texts)
        item["confidence"] = round(sum(_conf(l) for l in row) / len(row), 3)
        items.append(item)
    return items
