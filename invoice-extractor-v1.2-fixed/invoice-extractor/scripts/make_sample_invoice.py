"""Generate a synthetic invoice PNG for demos and parser tests.

No real company data — safe to commit, share, and screenshot for a portfolio.

Usage:
    python scripts/make_sample_invoice.py --out samples/invoice_001.png
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_invoice(path: str) -> None:
    W, H = 1240, 1754  # ~A4 at 150 DPI — upscale target for the pipeline
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text((80, 60), "ACME TRADERS", font=_font(64), fill="black")
    d.text((80, 140), "42 Market Road, Lahore, Pakistan", font=_font(28), fill="black")
    d.text((80, 180), "Phone: +92-300-1234567", font=_font(28), fill="black")

    d.text((80, 300), "Invoice No: INV-2026-0481", font=_font(36), fill="black")
    d.text((80, 350), "Invoice Date: 15 Sep 2026", font=_font(36), fill="black")
    d.text((80, 400), "Currency: PKR", font=_font(36), fill="black")

    # Table header
    y = 520
    d.text((80, y), "Description", font=_font(34), fill="black")
    d.text((640, y), "Qty", font=_font(34), fill="black")
    d.text((780, y), "Unit Price", font=_font(34), fill="black")
    d.text((1000, y), "Amount", font=_font(34), fill="black")
    d.line((80, y + 50, 1160, y + 50), fill="black", width=3)

    rows = [
        ("Widget A - premium grade", "2", "500.00", "1000.00"),
        ("Widget B - standard", "5", "120.00", "600.00"),
        ("Service charges", "1", "150.00", "150.00"),
    ]
    y += 70
    for desc, qty, price, amount in rows:
        d.text((80, y), desc, font=_font(30), fill="black")
        d.text((640, y), qty, font=_font(30), fill="black")
        d.text((780, y), price, font=_font(30), fill="black")
        d.text((1000, y), amount, font=_font(30), fill="black")
        y += 60

    y += 40
    d.text((780, y), "Subtotal:", font=_font(32), fill="black")
    d.text((1000, y), "1750.00", font=_font(32), fill="black")
    y += 55
    d.text((780, y), "Tax (16%):", font=_font(32), fill="black")
    d.text((1000, y), "280.00", font=_font(32), fill="black")
    y += 55
    d.text((780, y), "Grand Total:", font=_font(38), fill="black")
    d.text((1000, y), "2030.00", font=_font(38), fill="black")

    img.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/invoice_001.png")
    args = ap.parse_args()
    make_invoice(args.out)
