"""Image preprocessing pipeline (pure functions, OpenCV).

Ranked by measured impact (research):
  1. Rasterize PDFs at 300 DPI            <- single biggest factor
  2. Orientation correction               <- handled inside PaddleOCR
  3. Deskew (2 degrees breaks segmentation)
  4. Adaptive/Sauvola binarization
  5. Denoise
  6. Never upscale beyond source resolution.
"""
from __future__ import annotations

import io

import cv2
import numpy as np

try:
    import pypdfium2 as pdfium  # BSD-licensed PDF rasterizer
    _PDFIUM_OK = True
except ImportError:  # pragma: no cover
    _PDFIUM_OK = False


_MAX_RENDER_PX = 12000  # longest edge at render time; bigger -> OOM risk


# --------------------------------------------------------------------------- #
# PDF -> images
# --------------------------------------------------------------------------- #
def pdf_to_images(pdf_bytes: bytes, dpi: int = 300, max_pages: int = 20) -> list[np.ndarray]:
    """Rasterize each PDF page to a BGR numpy image at the requested DPI.

    Pages beyond max_pages are REJECTED with an error (never silently
    dropped). Pages with a huge MediaBox are rendered at a reduced scale
    instead of becoming a 300-DPI memory bomb.
    """
    if not _PDFIUM_OK:
        raise RuntimeError(
            "pypdfium2 is not installed — PDF input needs it (pip install pypdfium2)."
        )
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    n_pages = len(doc)
    if n_pages > max_pages:
        raise ValueError(
            f"PDF has {n_pages} pages; the limit is {max_pages} — "
            "refusing to silently drop pages. Split the file or raise MAX_PAGES."
        )
    images: list[np.ndarray] = []
    for page in doc:
        try:
            x0, y0, x1, y1 = (float(v) for v in page.get_mediabox()[:4])
        except Exception:
            x0, y0, x1, y1 = 0.0, 0.0, 595.0, 842.0  # A4 fallback (points)
        scale = dpi / 72.0
        if max(x1 - x0, y1 - y0) * scale > _MAX_RENDER_PX:
            scale = _MAX_RENDER_PX / max(x1 - x0, y1 - y0)
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        images.append(cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR))
    return images


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate text skew via minimum-area rectangle of foreground pixels."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns [-90, 0); normalize to a small deskew angle.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def deskew(gray: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """Sauvola-style adaptive threshold for uneven lighting (receipts/photos)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def preprocess(image: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """Full pipeline. Keep it minimal for clean scans (research: minimal beats
    heavy enhancement on ~95% of typical documents); use aggressive=True only
    for severely degraded photos.

    Always returns 3-channel BGR: warmup and PaddleOCR predict() both feed
    3-channel images, so the pipeline must too.
    """
    gray = to_grayscale(image)
    gray = deskew(gray)
    gray = denoise(gray)
    if aggressive:
        gray = adaptive_binarize(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ensure_min_dpi(image: np.ndarray, min_long_edge: int = 2200) -> np.ndarray:
    """Upscale ONLY if the image is smaller than a ~300-DPI A4 scan.
    Never invent pixels beyond a sane target."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge >= min_long_edge:
        return image
    scale = min_long_edge / long_edge
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
