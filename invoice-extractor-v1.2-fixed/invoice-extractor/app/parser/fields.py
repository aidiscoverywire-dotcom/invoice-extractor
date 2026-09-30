"""Rule-based invoice field extraction (stdlib only — no ML, no heavy deps).

Strategy (from research):
  - Anchored regexes per field; VAT/amount validation; character-confusion repair
    in numeric fields (O<->0, l/I<->1, S<->5, B<->8 — the dominant OCR error class).
  - Total via LABEL RANKING ("grand total" > "amount due" > "total"), never the
    largest-amount heuristic.
  - Vendor from the top-of-page largest text region.
  - Cross-field arithmetic: subtotal + tax == total catches a wrong digit.

Input: list of dicts {text, confidence, bbox:{x_min..}|None, page}.
Output: dict of {"value","confidence","source"} per field + arithmetic_ok flag.
"""
from __future__ import annotations

import re
from datetime import datetime

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _text(line) -> str:
    return line["text"] if isinstance(line, dict) else line.text


def _conf(line) -> float:
    c = line.get("confidence", 0.0) if isinstance(line, dict) else line.confidence
    return float(c or 0.0)


def _bbox(line):
    b = line.get("bbox") if isinstance(line, dict) else line.bbox
    if b is None:
        return None
    return b if isinstance(b, dict) else b.model_dump()


def _page(line) -> int:
    p = line.get("page", 1) if isinstance(line, dict) else getattr(line, "page", 1)
    try:
        return int(p or 1)
    except (TypeError, ValueError):
        return 1


# Words that mark a table header row — used to keep header rows out of
# total/tax searches (a "Total" column header must not donate its value).
_HEADER_WORDS = {"description", "item", "particulars", "qty", "quantity",
                 "price", "rate", "unit", "amount", "total", "pcs"}
_QTY_RE = re.compile(r"\bqty\b|\bquantity\b", re.IGNORECASE)


def _looks_like_header(text: str) -> bool:
    return len(set(text.lower().split()) & _HEADER_WORDS) >= 2


def _lbl(phrase: str) -> re.Pattern:
    """Label matcher with real word boundaries: 'total amount' won't match
    'subtotal amounts', and 'dated' won't match 'updated'."""
    return re.compile(r"\b" + phrase.replace(" ", r"\s+") + r"\b", re.IGNORECASE)


# Common OCR character confusions in NUMERIC contexts.
_NUMERIC_REPAIRS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                                  "S": "5", "s": "5", "B": "8", "b": "6"})


def repair_numeric(raw: str) -> str:
    """Repair OCR character confusions inside numeric-looking TOKENS only.

    'Rs 500' -> 'Rs 500' (currency prefix untouched), '1,O00' -> '1,000'.
    A token is repaired only if it contains a digit and consists solely of
    digits, separators and confusion characters.
    """
    def fix_token(tok: str) -> str:
        if re.search(r"\d", tok) and re.fullmatch(r"[OolISsBb0-9,.\-\s%$€£]+", tok):
            return tok.translate(_NUMERIC_REPAIRS)
        return tok
    return " ".join(fix_token(t) for t in raw.split(" "))


# Digit class tolerant to the dominant OCR confusions (O<->0, l/I<->1, S<->5, B<->8).
# The currency prefix is matched FIRST on the raw string, so "Rs" never gets
# corrupted — confusions are repaired only inside the digit run itself.
_DIG = r"[0-9OoIlISsBb]"
_AMOUNT_RE = re.compile(
    r"(?:Rs\.?|PKR|USD|EUR|GBP|\$|€|£)?\s*"
    # Optional accounting negative between a currency prefix and the digits,
    # so "Rs -500.00" / "-500.00" / "(500.00)" all parse (it stays OUTSIDE
    # group(1) — _parse_match reads the sign from the match prefix instead).
    r"\(?[-\u2212]?\s*"
    r"("
    # Thousands separators: US/UK "1,250" / "1,250.00" ...
    rf"{_DIG}{{1,3}}(?:,{_DIG}{{3}})+(?:\.{_DIG}{{2}})?"
    # ... and EU "2.030" / "2.030,00" / "1.234.567,89".
    rf"|{_DIG}{{1,3}}(?:\.{_DIG}{{3}})+(?:,{_DIG}{{1,2}})?"
    rf"|{_DIG}+\.{_DIG}{{2}}"      # 2030.00
    rf"|{_DIG}+,{_DIG}{{1,2}}"     # 12,50 (EU decimal comma)
    rf"|{_DIG}+"                   # 2030
    r")"
)


def _clean_number(raw: str) -> float | None:
    """'2.030,00' -> 2030.0 (EU), '1,250.00' -> 1250.0 (US/UK), '12,50' -> 12.5.

    Separator rule: when both '.' and ',' appear, the RIGHTMOST one is the
    decimal separator ("2.030,00" vs "1,250.00"). Comma-only: "1,250"/"1,250,000"
    are thousands, "12,50"/"1,5" a decimal comma, anything else (e.g. Indian
    grouping "1,25,000") falls back to thousands. Dot-only stays US/UK decimal —
    a lone "2.030" reads as 2.03 (ambiguous, documented in the README).
    """
    if not re.search(r"[0-9]", raw):  # must contain at least one real digit
        return None
    s = raw.translate(_NUMERIC_REPAIRS)
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")
    if last_comma != -1 and last_dot != -1:
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif last_comma != -1:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            s = "".join(parts)
        elif len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
            s = parts[0] + "." + parts[1]
        else:
            s = "".join(parts)
    else:
        # Dot only: "1.234.567" (2+ dot groups) is unambiguously EU thousands.
        # A lone "2.030" stays the US decimal 2.03 — genuinely ambiguous,
        # documented in the README.
        dot_groups = s.split(".")
        if len(dot_groups) > 2 and all(g.isdigit() and len(g) == 3
                                       for g in dot_groups[1:]):
            s = "".join(dot_groups)
    try:
        return float(s)
    except ValueError:
        return None


def _is_negative(prefix: str, suffix: str) -> bool:
    """Accounting negatives: "(500.00)" / "($1,250.00)" or "-500.00" / "Rs -500.00".

    A '-' counts only when it is NOT glued to a word character — "Invoice-2026"
    keeps its hyphen, while "-500" and "Rs -500" are negative amounts.
    """
    if "(" in prefix and ")" in suffix:
        return True
    m = re.search(r"[-\u2212]\s*$", prefix)
    if not m:
        return False
    i = m.start()
    return i == 0 or prefix[i - 1].isspace()


def _parse_match(raw: str, m: re.Match) -> float | None:
    val = _clean_number(m.group(1))
    if val is None:
        return None
    if val != 0 and _is_negative(raw[:m.start(1)], raw[m.end(1):]):
        val = -val
    return val


def parse_amount(raw: str) -> float | None:
    """Parse 'Rs 1,250.00' / '$1,250.00' / '1,O00.00' (OCR-mangled) -> float.

    Also handles negatives ('(500.00)', '-500.00') and EU formatting
    ('2.030,00' -> 2030.0).
    """
    m = _AMOUNT_RE.search(raw)
    if not m:
        return None
    return _parse_match(raw, m)


def _line_amounts(text: str) -> list[float]:
    # A bare "16%" is never an amount — strip percentages first so
    # "Tax (16%):" doesn't yield 16.0 instead of the next line's 280.00.
    text = re.sub(r"\(?\d+(?:\.\d+)?\s*%\)?", "", text)
    out: list[float] = []
    for m in _AMOUNT_RE.finditer(text):
        v = _parse_match(text, m)
        if v is not None:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Invoice number
# --------------------------------------------------------------------------- #
_INV_PATTERNS = [
    # Labeled with a number-word — "Invoice No:", "Invoice #". The (no|number|#)
    # part is REQUIRED so "Invoice Date:" / "Invoice Total:" never match.
    re.compile(r"(?:invoice|inv)[\s.\-]*(?:no\.?|number|num|#)\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-/]{1,})", re.IGNORECASE),
    # Labeled with a bare colon — "Invoice: INV-2026-0481" (but not "Invoice Date:").
    re.compile(r"(?:invoice|inv)\s*[:#]\s*([A-Z0-9][A-Z0-9\-/]{1,})", re.IGNORECASE),
    # Bare token — must start with INV and contain a digit.
    re.compile(r"\b(INV[\-/]?\d[\d\-/]*)\b", re.IGNORECASE),
]
_INV_STOPWORDS = {"date", "total", "due", "tax", "no", "number"}


def _valid_inv_token(token: str) -> bool:
    t = token.strip("-/ ")
    return bool(re.search(r"\d", t)) and t.lower() not in _INV_STOPWORDS and len(t) >= 3


def find_invoice_number(lines: list) -> dict:
    for line in lines:
        t = _text(line)
        for pat in _INV_PATTERNS:
            m = pat.search(t)
            if m and _valid_inv_token(m.group(1)):
                # Invoice numbers are conventionally uppercase.
                return {"value": m.group(1).strip("-/ ").upper(), "confidence": _conf(line),
                        "source": "regex:invoice_number", "bbox": _bbox(line)}
    return {"value": None, "confidence": 0.0, "source": "", "bbox": None}


# --------------------------------------------------------------------------- #
# Dates -> ISO YYYY-MM-DD
# --------------------------------------------------------------------------- #
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

_DATE_RES = [
    (re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"), "dmy"),
    (re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b"), "dmy_text"),
    (re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"), "mdy_text"),
    # 2-digit year: "15/09/26" -> 2026-09-15.
    (re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b"), "dmy2"),
]

_DATE_LABEL_RES = [
    re.compile(r"\binvoice\s+date\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+issue\b", re.IGNORECASE),
    # (?<!\w) so "dated" matches "Dated:" but NOT "updated".
    re.compile(r"(?<!\w)dated\b", re.IGNORECASE),
    re.compile(r"\bbill\s+date\b", re.IGNORECASE),
]
# Lines that look like dates but aren't the invoice date.
_DATE_EXCLUDE_RES = [
    re.compile(r"\bdue\b", re.IGNORECASE),      # "Due Date:" is never the invoice date
    re.compile(r"\bupdated\b", re.IGNORECASE),   # "Last updated:" is metadata
]


def _to_iso(day: int, month: int, year: int) -> str | None:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _year2(yy: int) -> int:
    """2-digit year pivot for invoices: 26 -> 2026, 99 -> 1999."""
    return 2000 + yy if yy <= 30 else 1900 + yy


def _parse_numeric_date(raw: str, first: int, second: int, year: int,
                        day_first: bool) -> tuple[str | None, str | None]:
    """Parse DD/MM/YYYY vs MM/DD/YYYY. Returns (iso, warning-note).

    Unambiguous dates (either part > 12) parse silently. Ambiguous ones
    ("05/06/2026") parse per `day_first` and ALWAYS carry a warning note so a
    human can verify — a US invoice's May 6 must never silently become June 5.
    """
    d, mo = (first, second) if day_first else (second, first)
    iso = _to_iso(d, mo, year) or _to_iso(mo, d, year)
    note = None
    if iso and first != second and 1 <= first <= 12 and 1 <= second <= 12:
        basis = "day-first" if day_first else "month-first"
        note = (f"ambiguous numeric date '{raw}' parsed {basis} as {iso}; "
                f"verify against the source document")
    return iso, note


def find_invoice_date(lines: list, day_first: bool = True) -> dict:
    """day_first: how to read ambiguous numeric dates ("05/06/2026").
    Ambiguous dates always carry a "note" warning — see _parse_numeric_date."""
    # Prefer lines carrying a date label; fall back to any date-like line.
    # "Due Date:" and "Last updated:" lines are excluded — they are never
    # the invoice date.
    def labeled(t): return any(r.search(t) for r in _DATE_LABEL_RES)
    def excluded(t): return any(r.search(t) for r in _DATE_EXCLUDE_RES)

    cands = [l for l in lines if not excluded(_text(l))]
    ordered = sorted(cands, key=lambda ln: 0 if labeled(_text(ln)) else 1)
    for line in ordered:
        t = _text(line)
        for pat, kind in _DATE_RES:
            m = pat.search(t)
            if not m:
                continue
            g = m.groups()
            iso = None
            note = None
            if kind == "dmy":
                iso, note = _parse_numeric_date(
                    m.group(0), int(g[0]), int(g[1]), int(g[2]), day_first)
            elif kind == "ymd":
                iso = _to_iso(int(g[2]), int(g[1]), int(g[0]))
            elif kind == "dmy2":
                iso, note = _parse_numeric_date(
                    m.group(0), int(g[0]), int(g[1]), _year2(int(g[2])), day_first)
            elif kind == "dmy_text":
                mo = _MONTHS.get(g[1][:3].lower())
                iso = _to_iso(int(g[0]), mo, int(g[2])) if mo else None
            elif kind == "mdy_text":
                mo = _MONTHS.get(g[0][:3].lower())
                iso = _to_iso(int(g[1]), mo, int(g[2])) if mo else None
            if iso:
                return {"value": iso, "confidence": _conf(line),
                        "source": "regex:date", "bbox": _bbox(line),
                        "note": note}
    return {"value": None, "confidence": 0.0, "source": "", "bbox": None,
            "note": None}


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #
# \b fails on "$"/"€"/"£" (symbol followed by a digit has no word boundary),
# so word-tokens and symbol-tokens get different boundary rules.
_CURRENCY_RE = re.compile(
    r"(?<!\w)(Rs\.?|PKR|USD|EUR|GBP)(?![A-Za-z])|(\$|€|£)", re.IGNORECASE)
_CURRENCY_NORM = {"rs": "PKR", "rs.": "PKR", "pkr": "PKR", "$": "USD",
                  "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP"}


def find_currency(lines: list) -> dict:
    # Currency tokens are cheap to match, so scan every line — the old
    # first-40-lines cutoff missed declarations buried in long documents.
    for line in lines:
        m = _CURRENCY_RE.search(_text(line))
        if m:
            tok = m.group(1) or m.group(2)
            code = _CURRENCY_NORM.get(tok.lower(), tok.upper())
            return {"value": code, "confidence": _conf(line),
                    "source": "regex:currency", "bbox": _bbox(line)}
    return {"value": None, "confidence": 0.0, "source": "", "bbox": None}


# --------------------------------------------------------------------------- #
# Totals with LABEL RANKING (never the largest-amount heuristic).
# Priority order is real: labels are tried across ALL lines, so "Grand Total"
# wins over "Net Total" even when the net-total line comes first.
# --------------------------------------------------------------------------- #
_TOTAL_LABEL_RES = [_lbl("grand total"), _lbl("amount due"), _lbl("balance due"),
                    _lbl("total due"), _lbl("net total"), _lbl("total amount"),
                    _lbl("invoice total")]
_PLAIN_TOTAL = re.compile(r"(?<!\w)total(?!\w)", re.IGNORECASE)
_SUBTOTAL_LABEL_RES = [_lbl("subtotal"), _lbl("sub total"), _lbl("sub-total")]
_TAX_LABEL_RES = [re.compile(r"\bvat\b", re.IGNORECASE),
                  re.compile(r"\bgst\b", re.IGNORECASE),
                  re.compile(r"\bsales\s+tax\b", re.IGNORECASE),
                  re.compile(r"\bhst\b", re.IGNORECASE),
                  _lbl("tax")]
# "TAX INVOICE" / "Tax ID:" are document metadata, never the tax amount.
_TAX_EXCLUDE = re.compile(r"\btax\s+(invoice|id|no\.?|number)\b", re.IGNORECASE)


def _candidate_lines_for_label(lines: list, label_re, exclude_re=None) -> list[int]:
    idxs = []
    for idx, line in enumerate(lines):
        t = _text(line)
        if exclude_re and exclude_re.search(t):
            continue
        if _looks_like_header(t):   # a "Total" column header donates nothing
            continue
        if _QTY_RE.search(t):       # "Total Qty: 7" is not the invoice total
            continue
        if label_re.search(t):
            idxs.append(idx)
    return idxs


def _amount_near_label(lines: list, label_res: list, plain_re=None,
                       exclude_re=None) -> dict:
    """Amount on the label's line, else the nearest line below it (2-line
    lookahead). Returns the source line's bbox for the review queue."""
    def scan(idxs: list[int]) -> dict | None:
        for idx in idxs:
            for cand in (lines[idx], *lines[idx + 1: idx + 3]):
                if _looks_like_header(_text(cand)):
                    continue
                amounts = _line_amounts(_text(cand))
                if amounts:
                    return {"value": amounts[-1], "confidence": _conf(cand),
                            "source": "regex:label_ranked", "bbox": _bbox(cand)}
        return None

    for lab in label_res:
        hit = scan(_candidate_lines_for_label(lines, lab, exclude_re))
        if hit:
            return hit
    if plain_re is not None:
        hit = scan(_candidate_lines_for_label(lines, plain_re, exclude_re))
        if hit:
            return hit
    return {"value": None, "confidence": 0.0, "source": "", "bbox": None}


def find_totals(lines: list) -> dict:
    total = _amount_near_label(lines, _TOTAL_LABEL_RES, _PLAIN_TOTAL)
    subtotal = _amount_near_label(lines, _SUBTOTAL_LABEL_RES)
    tax = _amount_near_label(lines, _TAX_LABEL_RES, exclude_re=_TAX_EXCLUDE)
    return {"subtotal": subtotal, "tax": tax, "total": total}


# --------------------------------------------------------------------------- #
# Vendor — top-of-page largest text region
# --------------------------------------------------------------------------- #
def find_vendor(lines: list) -> dict:
    if not lines:
        return {"value": None, "confidence": 0.0, "source": "", "bbox": None}
    # Vendor lives on the FIRST page — never mix pages.
    first_page = min(_page(l) for l in lines)
    lines = [l for l in lines if _page(l) == first_page]
    if not lines:
        return {"value": None, "confidence": 0.0, "source": "", "bbox": None}
    boxes = [(_bbox(l), l) for l in lines]
    with_box = [(b, l) for b, l in boxes if b]
    if with_box:
        max_y = max(b["y_max"] for b, _ in with_box)
        top = [(b, l) for b, l in with_box if b["y_min"] <= max_y * 0.25]
        cands = top or with_box
        # largest area among alphabetic lines
        best, best_area = None, -1.0
        for b, l in cands:
            t = _text(l)
            if not re.search(r"[A-Za-z]{3,}", t):
                continue
            area = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            if area > best_area:
                best, best_area = l, area
        if best is not None:
            return {"value": _text(best).strip(), "confidence": _conf(best),
                    "source": "layout:top_region_largest"}
    # Fallback: first alphabetic line of the document.
    for line in lines[:8]:
        t = _text(line).strip()
        if re.search(r"[A-Za-z]{3,}", t) and len(t) <= 80:
            return {"value": t, "confidence": _conf(line), "source": "fallback:first_line"}
    return {"value": None, "confidence": 0.0, "source": ""}


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def extract_fields(lines: list, day_first: bool = True) -> dict:
    """lines: list of {text, confidence, bbox|None, page} dicts (reading order).

    day_first: passed to find_invoice_date for ambiguous numeric dates.
    """
    totals = find_totals(lines)
    return {
        "vendor": find_vendor(lines),
        "invoice_number": find_invoice_number(lines),
        "invoice_date": find_invoice_date(lines, day_first=day_first),
        "currency": find_currency(lines),
        "subtotal": totals["subtotal"],
        "tax": totals["tax"],
        "total": totals["total"],
    }
