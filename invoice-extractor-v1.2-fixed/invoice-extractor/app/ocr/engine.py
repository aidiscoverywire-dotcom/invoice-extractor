"""PaddleOCR 3.x (PP-OCRv5) wrapper.

- Lazy singleton: models load once, on first use (or via warmup() at boot).
- Defensive output parsing: handles the 3.x `predict()` dict format
  ({rec_texts, rec_scores, rec_polys}) and falls back to the legacy 2.x
  list format, so minor version drift doesn't silently break extraction.
- Always returns normalized OcrLine dicts: {text, confidence, bbox, page}.

Written against PaddleOCR 3.x. If you upgrade the paddleocr package, re-check
the official docs for output-format changes and extend _normalize().
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

from app.core.config import settings
from app.models.schemas import BBox, OcrLine

log = logging.getLogger(__name__)

_lock = threading.Lock()
_infer_lock = threading.Lock()  # PaddlePaddle inference is NOT thread-safe
_engine = None


def _poly_to_bbox(poly) -> BBox | None:
    try:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return BBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))
    except Exception:
        return None


def _as_list(v) -> list:
    """rec_scores etc. may be numpy arrays — `v or []` would crash with
    'truth value of an array is ambiguous'."""
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _normalize_predict_output(result, page: int) -> list[OcrLine]:
    """Normalize PaddleOCR 3.x predict() output -> list[OcrLine]."""
    lines: list[OcrLine] = []
    items = result if isinstance(result, list) else [result]
    for res in items:
        if not isinstance(res, dict):
            continue
        texts = _as_list(res.get("rec_texts"))
        scores = _as_list(res.get("rec_scores"))
        polys = _as_list(res.get("rec_polys"))
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            conf = float(scores[i]) if i < len(scores) else 0.0
            bbox = _poly_to_bbox(polys[i]) if i < len(polys) else None
            lines.append(OcrLine(text=text, confidence=conf, bbox=bbox, page=page))
    return lines


def _normalize_legacy_output(result, page: int) -> list[OcrLine]:
    """PaddleOCR 2.x .ocr() format: [[ [box], (text, score) ], ...]."""
    lines: list[OcrLine] = []
    for entry in result or []:
        try:
            box, (text, score) = entry
            text = (text or "").strip()
            if not text:
                continue
            lines.append(OcrLine(text=text, confidence=float(score),
                                 bbox=_poly_to_bbox(box), page=page))
        except Exception:
            continue
    return lines


class OcrEngine:
    """Thread-safe lazy PaddleOCR singleton."""

    def __init__(self) -> None:
        self._ocr = None
        self._ready = False

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed. Install the CPU stack first:\n"
                '  pip install "paddlepaddle==3.2.1" '
                "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
                "  pip install -r requirements.txt"
            ) from exc

        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "HF")
        # PaddleX (the PaddleOCR 3.x backend) caches models under
        # $PADDLE_PDX_CACHE_HOME (default ~/.paddlex); older docs also mention
        # $PADDLEX_HOME. Set both to the configured model dir so the cache
        # lands in one predictable place regardless of version.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", settings.model_dir)
        os.environ.setdefault("PADDLEX_HOME", settings.model_dir)
        log.info("Loading PaddleOCR (lang=%s, device=%s)…", settings.ocr_lang, settings.ocr_device)
        self._ocr = PaddleOCR(
            lang=settings.ocr_lang,
            device=settings.ocr_device,
        )
        self._ready = True
        log.info("PaddleOCR ready.")

    def warmup(self) -> None:
        """Call at app boot: absorbs the 8–12 s cold start before traffic."""
        with _lock:
            if not self._ready:
                self._load()
        # One dummy pass so lazy sub-models initialize now, not mid-request.
        blank = np.full((64, 256, 3), 255, dtype=np.uint8)
        try:
            self.ocr_image(blank, page=1)
        except Exception:
            log.warning("Warmup pass produced no output (harmless on blank image).")

    def is_ready(self) -> bool:
        return self._ready

    def ocr_image(self, image: np.ndarray, page: int = 1) -> list[OcrLine]:
        with _lock:
            if not self._ready:
                self._load()
            ocr = self._ocr

        # PaddlePaddle inference is not thread-safe: serialize it. The
        # /extract route runs in FastAPI's threadpool (plain `def`), so
        # concurrent requests queue here instead of corrupting the interpreter.
        with _infer_lock:
            # Prefer the 3.x unified predict() API; legacy 2.x ocr() only when
            # predict() doesn't exist at all. An EMPTY predict() result is a
            # valid answer (blank page) — falling through to ocr() would run
            # OCR twice and double the latency.
            if hasattr(ocr, "predict"):
                return _normalize_predict_output(ocr.predict(image), page)
            if hasattr(ocr, "ocr"):
                return _normalize_legacy_output(ocr.ocr(image), page)
        raise RuntimeError("Unrecognized PaddleOCR API — check installed version docs.")


def get_engine() -> OcrEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = OcrEngine()
    return _engine
