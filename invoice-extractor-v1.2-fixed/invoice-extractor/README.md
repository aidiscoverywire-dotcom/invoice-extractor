# Invoice Extractor

Open-source, **zero-cost** invoice & receipt data extraction API.
Upload a PDF or image → get structured JSON: vendor, invoice number, dates,
line items, subtotal, tax, total — with per-field confidence scores and a
human review queue for uncertain fields.

Built from a [deep research pass](../research_notes/ocr-invoice-extraction/RESEARCH_SUMMARY.md)
so every stack choice is evidence-backed, not guessed.

## Stack (all Apache-2.0 / MIT — commercial-safe, $0 cost)

| Layer | Choice | Why |
|---|---|---|
| OCR | **PaddleOCR 3.x (PP-OCRv5)** | Best CPU accuracy/speed balance (~23 pg/min), per-line confidence, native PDF input |
| Fallback OCR | Tesseract 5.5 (optional) | Fastest engine; used only on low-confidence regions |
| Preprocessing | OpenCV | 300 DPI raster, deskew, denoise, adaptive threshold (opt-in) |
| PDF raster | pypdfium2 | BSD-licensed (PyMuPDF is AGPL — avoided) |
| Field parsing | Rules + regex | Deterministic, zero training data, millisecond runtime |
| Validation | Arithmetic cross-check | `subtotal + tax == total` catches single-digit errors |
| API | FastAPI | Async-ready, auto docs at `/docs` |
| DB | PostgreSQL (SQLite for local dev) | Documents, review queue, correction history |
| Deploy | Docker, x86_64 CPU | Pre-baked models, warmup on boot |

> ⚠️ **Deliberately NOT used:** Surya OCR (GPL-3.0 + revenue-threshold weights — a licensing trap
> for client products), Donut (needs GPU), paid vision APIs (violates the zero-cost rule).

## Quickstart

### With Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
# API → http://localhost:8000/docs
```

First boot downloads OCR models (~tens of MB) and warms up (~10 s). Models are
baked into the image on build via `scripts/download_models.py`, so runtime is offline-safe.

### Local dev (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
# 1. CPU-only PaddlePaddle first (from the official CPU index):
pip install "paddlepaddle==3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# 2. Everything else:
pip install -r requirements.txt
# 3. Pre-download OCR models:
python scripts/download_models.py
# 4. Run (SQLite by default — no Postgres needed):
uvicorn app.main:app --reload
```

## Try it

```bash
# 1. Generate a sample invoice image:
python scripts/make_sample_invoice.py --out samples/invoice_001.png

# 2. Extract:
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@samples/invoice_001.png" | python -m json.tool
```

Response (abridged):

```json
{
  "document_id": "a3f1…",
  "status": "needs_review",
  "fields": {
    "vendor": {"value": "Acme Traders", "confidence": 0.97},
    "invoice_number": {"value": "INV-2026-0481", "confidence": 0.99},
    "invoice_date": {"value": "2026-09-15", "confidence": 0.93},
    "total": {"value": 1250.00, "confidence": 0.88}
  },
  "line_items": [{"description": "Widget A", "qty": 2, "unit_price": 500.0, "amount": 1000.0}],
  "low_confidence_fields": ["total"]
}
```

## API

**Auth (optional but recommended):** set `API_KEY` in `.env` — every `/api/v1/*`
route then requires the `X-API-Key` header (compared in constant time).
With no `API_KEY` the API is open (local dev only). Auth'd routes are
rate-limited per key: `RATE_LIMIT_PER_MINUTE` (default 120) requests per
60-second window; excess calls get HTTP `429`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + OCR warmup state (always public) |
| `POST` | `/api/v1/extract` | Upload PDF/PNG/JPG → structured extraction (`?force=true` reprocesses even a cached file) |
| `GET` | `/api/v1/documents` | List documents (`?status=auto/needs_review/quarantined/verified`) |
| `GET` | `/api/v1/documents/{id}` | Full extraction incl. OCR lines + bboxes. Paginated: `?include_ocr_lines=false` skips the lines payload, `?ocr_lines_limit=2000&ocr_lines_offset=0` pages it (`ocr_lines_total` always reports the full count) |
| `GET` | `/api/v1/review/pending` | Human review queue |
| `POST` | `/api/v1/review/{item_id}/correct` | Submit a correction (append-only history; updates the document's stored fields; last pending correction flips the document to `verified`). Corrections are type-validated — e.g. a money field rejects `"N/A"` with HTTP `422` |

**Duplicate uploads:** same file bytes (sha256) are processed once — repeats return
the cached result unless `?force=true`. Simultaneous duplicate uploads are
serialized safely (integrity fallback). Oversized uploads are rejected while
streaming; PDFs over `MAX_PAGES` (default 20) are rejected, not truncated.

**Status semantics:** a document is `needs_review` when any critical field is
weak, when its average confidence is mid-band, or when `subtotal + tax ≠ total`
(arithmetic mismatch escalates even high-confidence results). Corrections are
per-field with append-only history.

Interactive docs: `http://localhost:8000/docs`

## How it works

```
upload → sha256 dedupe → PDF→300DPI raster (pypdfium2)
  → OpenCV preprocess (deskew → denoise → *optional* adaptive threshold)
  → PaddleOCR PP-OCRv5 (text + confidence + boxes per line)
  → [optional] Tesseract fallback on low-confidence regions
  → field extraction (regex + label-ranked totals + confusion repair)
  → line-item table detection (geometric row/column clustering)
  → arithmetic validation (subtotal + tax == total)
  → confidence gate → auto / needs_review / quarantined
  → persist + review queue
```

### Confidence gates (configurable in `.env`)

| Band | Decision |
|---|---|
| avg ≥ `0.92` and critical fields ≥ `0.80` | `auto` — accepted |
| avg `0.75`–`0.92` or any critical field weak | `needs_review` — human checks |
| avg < `0.75` | `quarantined` — re-capture |

Thresholds are **not** ground truth — calibrate them on your own labeled data
(see `eval/`). Scores are not comparable across OCR engines.

### Number & date formats

Amounts accept US (`2,030.00`), European (`2.030,00`), and plain (`2030.00`)
forms, plus accounting negatives (`(500.00)`, `-500.00`). A lone ambiguous
`2.030` is read as US `2.03` — if your invoices are European, prefer the
explicit `2.030,00` form. Ambiguous numeric dates (`05/06/2026`) follow
`DATE_DAY_FIRST` (default `true` = 5 Jun 2026) and always carry a warning
note on the document so a human verifies the interpretation.

## Project structure

```
invoice-extractor/
├── app/
│   ├── main.py            # FastAPI app + lifespan warmup
│   ├── pipeline.py        # end-to-end orchestration
│   ├── api/routes.py      # HTTP endpoints
│   ├── core/config.py     # settings (.env)
│   ├── db/                # SQLAlchemy models + session
│   ├── models/schemas.py  # Pydantic request/response models
│   ├── ocr/
│   │   ├── preprocess.py  # OpenCV pipeline (pure functions)
│   │   ├── engine.py      # PaddleOCR wrapper (lazy, defensive output parsing)
│   │   └── fallback.py    # optional Tesseract fallback
│   ├── parser/
│   │   ├── fields.py      # regex field extraction (stdlib only)
│   │   ├── tables.py      # geometric line-item clustering (stdlib only)
│   │   └── validate.py    # arithmetic checks + confidence gating
│   └── review/queue.py    # review queue helpers
├── scripts/
│   ├── download_models.py # pre-bake OCR weights
│   └── make_sample_invoice.py  # synthetic demo invoice (Pillow)
├── tests/                 # pytest — parser/table/gating run with stdlib only
├── eval/                  # field-level P/R/F1 + CER harness (skeleton + README)
├── samples/               # put real (anonymized) invoices here to try
├── Dockerfile
├── docker-compose.yml
└── THIRD_PARTY_LICENSES.md
```

## Evaluation

```bash
# predictions.json + ground_truth.json → field P/R/F1 and CER
python eval/evaluate.py --pred eval/predictions.json --gt eval/ground_truth.json
```

Per-document P/R/F1 plus **aggregate micro P/R/F1** (over all field
predictions), **macro F1** (mean of per-field F1), and per-field F1 are
printed. Numeric fields compare by value, so `2030`, `2030.0`, and
`"2030.00"` count as the same answer.

See `eval/README.md`. Standard public datasets: **SROIE 2019** (1,000 receipts),
**CORD**, **FATURA** (1,000 invoices). Build a 50–100 page ground-truth set of
*your* document class and re-measure after every pipeline change.

## Roadmap

- **PP-StructureV3** table mode for complex/borderless tables (not installed by
  default — add `paddleocr[doc-parser]` deliberately if you need it)
- **LayoutLMv3** fine-tune tier for high-churn vendor layouts (500+ labeled pages)
- Worker pool + Redis queue for batch throughput (current: synchronous, ~23 pg/min on CPU)

## License

This project's own code: **MIT**. Third-party components keep their own licenses —
all Apache-2.0/MIT/BSD/HPND except psycopg2-binary (LGPL, dynamic link),
commercial-safe for typical use. See `THIRD_PARTY_LICENSES.md`.
