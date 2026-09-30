# Third-Party Licenses

This project's own code is **MIT**. Every dependency below keeps its own
license — commercial-safe for typical SaaS/self-hosted use (no AGPL network
copyleft, no revenue thresholds). Note: psycopg2-binary is **LGPL** — dynamic
linking keeps your code unaffected, but re-confirm upstream LICENSE files
before a commercial release.

| Component | License | Notes |
|---|---|---|
| PaddleOCR / PP-OCRv5 / PP-Structure | Apache-2.0 | OCR + layout/table models |
| PaddlePaddle | Apache-2.0 | DL framework (CPU wheel) |
| Tesseract OCR | Apache-2.0 | Optional fallback engine |
| OpenCV | Apache-2.0 | Preprocessing |
| Pillow | HPND (MIT-like) | Image I/O, sample generator |
| pypdfium2 | BSD-3-Clause | PDF rasterization |
| pdfplumber (eval/optional) | MIT | Native-PDF table extraction |
| FastAPI / Uvicorn / Starlette | MIT | API server |
| SQLAlchemy / Alembic | MIT | ORM + migrations |
| psycopg2-binary | LGPL-3.0 (with exceptions) | Postgres driver — dynamic link, no copyleft on your code |
| pydantic / pydantic-settings | MIT | Settings + schemas |
| jiwer | Apache-2.0 | CER/WER metrics |
| pytest | MIT | Tests |
| LayoutLMv3 (roadmap, optional) | MIT | Future Tier-2 field model |
| invoice2data (reference only) | MIT | Rules/template architecture reference |

## Deliberately excluded

| Component | License | Reason |
|---|---|---|
| Surya OCR | GPL-3.0 + cc-by-nc-sa-4.0 weights ($5M revenue/funding cap) | Copyleft + commercial threshold — unsafe for client products |
| Marker | Apache-2.0 code + OpenRAIL-M weights ($5M cap) | Weight license threshold |
| PyMuPDF / PyMuPDF4LLM | AGPL-3.0 | Network copyleft — use pypdfium2 instead |
| Nougat weights | CC-BY-NC | Non-commercial |
