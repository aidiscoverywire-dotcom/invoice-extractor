"""Optional Tesseract fallback for weak PaddleOCR regions.

Design (fixed):
- tesseract_available() checks the actual tesseract BINARY
  (pytesseract.get_tesseract_version()), not just the Python package.
- Word boxes come from image_to_data with Output.DICT — no pandas needed.
- merge_weak_regions() replaces ONLY low-confidence PaddleOCR lines whose
  region the Tesseract words cover better. Reading order is preserved;
  nothing is globally re-sorted and no duplicate lines are emitted.

Disabled by default (ENABLE_TESSERACT_FALLBACK=false). Needs the system binary:
`apt install tesseract-ocr` (+ language packs, e.g. tesseract-ocr-eng).
"""
from __future__ import annotations

from app.models.schemas import BBox, OcrLine


def tesseract_available() -> bool:
    """True only if the `tesseract` binary itself is installed and runnable."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()  # raises if the binary is missing
        return True
    except Exception:
        return False


def tesseract_lines(image, page: int = 1, lang: str = "eng") -> list[OcrLine]:
    """Per-word OCR boxes via pytesseract dict output (stdlib + pytesseract)."""
    import pytesseract

    data = pytesseract.image_to_data(image, lang=lang,
                                     output_type=pytesseract.Output.DICT)
    lines: list[OcrLine] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i]) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        x, y = float(data["left"][i]), float(data["top"][i])
        w, h = float(data["width"][i]), float(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        lines.append(OcrLine(
            text=text,
            confidence=max(0.0, min(1.0, conf)),
            bbox=BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
            page=page,
        ))
    return lines


def _center_in(box: BBox, x: float, y: float) -> bool:
    return box.x_min <= x <= box.x_max and box.y_min <= y <= box.y_max


def merge_weak_regions(primary: list[OcrLine], fallback: list[OcrLine],
                       weak_below: float = 0.5) -> list[OcrLine]:
    """Replace weak primary lines with covering Tesseract words.

    Only lines with confidence < weak_below are candidates, and only when the
    Tesseract words inside that line's box read it with HIGHER mean
    confidence. Order and line count are unchanged — parsers that rely on
    "next line" lookahead keep working.
    """
    merged: list[OcrLine] = []
    for pl in primary:
        if pl.confidence >= weak_below or pl.bbox is None:
            merged.append(pl)
            continue
        cands = [f for f in fallback if f.bbox is not None and _center_in(
            pl.bbox, (f.bbox.x_min + f.bbox.x_max) / 2,
            (f.bbox.y_min + f.bbox.y_max) / 2)]
        if cands:
            mean_conf = sum(c.confidence for c in cands) / len(cands)
            if mean_conf > pl.confidence:
                cands.sort(key=lambda c: c.bbox.x_min)
                merged.append(OcrLine(
                    text=" ".join(c.text for c in cands),
                    confidence=round(mean_conf, 3),
                    bbox=pl.bbox,
                    page=pl.page,
                ))
                continue
        merged.append(pl)
    return merged
