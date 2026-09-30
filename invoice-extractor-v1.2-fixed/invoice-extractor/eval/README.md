# Evaluation

Measure the pipeline the way research says matters: **field-level
precision/recall/F1** (the business metric), plus CER/WER on raw OCR text.

## Ground truth format

`eval/ground_truth.json`:

```json
{"invoices": [
  {"id": "inv_001",
   "fields": {"vendor": "Acme Traders", "invoice_number": "INV-2026-0481",
              "invoice_date": "2026-09-15", "currency": "PKR",
              "subtotal": 1750.0, "tax": 280.0, "total": 2030.0},
   "line_items": [{"description": "Widget A", "qty": 2, "amount": 1000.0}]}
]}
```

## Running

```bash
# 1. Produce predictions for your labeled set (script it from /api/v1/extract):
python eval/evaluate.py --pred eval/predictions.json --gt eval/ground_truth.json
```

## Datasets

- **SROIE 2019** — 1,000 scanned receipts (company/date/address/total). The standard OCR+KIE benchmark.
- **CORD** — receipt key-value extraction.
- **FATURA** — 1,000 invoices, public invoice benchmark.

**Most important:** build a 50–100 page ground-truth set of *your own* document
class (the invoices you will actually process). Public datasets don't match
your vendors' layouts — calibrate thresholds on your data, and re-measure after
every pipeline change.
