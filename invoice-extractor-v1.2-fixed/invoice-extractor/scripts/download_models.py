"""Pre-download PaddleOCR model weights so Docker builds / deploys run offline.

Usage:
    python scripts/download_models.py [--lang en]

Set PADDLE_PDX_MODEL_SOURCE=BOS if HuggingFace is unreachable from your network.
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    # PaddleOCR 3.x (PaddleX) caches weights under PADDLE_PDX_CACHE_HOME
    # (default ~/.paddlex; some docs also mention PADDLEX_HOME). Pin both
    # before import so the Docker bake (ENV ...=/srv/models) and local runs
    # land in the same place.
    _home = os.environ.get("MODEL_DIR", "./models")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", _home)
    os.environ.setdefault("PADDLEX_HOME", _home)
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=os.environ.get("OCR_LANG", "en"))
    args = ap.parse_args()

    from paddleocr import PaddleOCR  # noqa: PLC0415

    print(f"[download_models] warming up PaddleOCR (lang={args.lang}) — "
          "this downloads weights on first run…")
    ocr = PaddleOCR(lang=args.lang, device="cpu")
    # Trigger every lazy sub-model now, not at request time.
    import numpy as np  # noqa: PLC0415
    blank = np.full((64, 256, 3), 255, dtype=np.uint8)
    try:
        ocr.predict(blank)
    except Exception:
        pass  # blank image may yield nothing; weights are cached regardless
    print("[download_models] done. Weights cached — the app can now run offline.")


if __name__ == "__main__":
    main()
