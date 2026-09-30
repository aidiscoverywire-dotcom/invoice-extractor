"""Field-level evaluation: predictions.json vs ground_truth.json.

Metrics (research):
  - Field-level precision / recall / F1  <- the business metric
  - CER/WER via jiwer on the concatenated OCR text
  - Arithmetic-consistency rate, review rate

Ground truth format:
    {"invoices": [{"id": "inv_001",
                   "fields": {"vendor": "Acme", "total": 2030.0, ...},
                   "line_items": [{"description": "...", "amount": 1000.0}]}]}
Predictions format: {"inv_001": {"fields": {...}, "line_items": [...]}, ...}

Usage:
    python eval/evaluate.py --pred eval/predictions.json --gt eval/ground_truth.json
"""
from __future__ import annotations

import argparse
import json


def _norm(v):
    """Normalize a GT/prediction value for comparison.

    Numbers compare NUMERICALLY regardless of Python type: 2030.0 (float),
    2030 (int) and "2030.00" (str) are all equal. Everything else compares
    as stripped lowercase strings. None/blank -> ("missing", "").
    """
    if v is None:
        return ("missing", "")
    if isinstance(v, bool):
        return ("str", str(v).lower())
    s = str(v).strip()
    if not s:
        return ("missing", "")
    try:
        return ("num", round(float(s.replace(",", "").replace(" ", "")), 2))
    except ValueError:
        return ("str", s.lower())


def field_scores(pred: dict, gt: dict, fields: list[str]) -> dict:
    tp = fp = fn = 0
    per_field = {}
    for f in fields:
        p, g = _norm(pred.get(f)), _norm(gt.get(f))
        if g[0] == "missing":
            continue  # field not labeled in GT — skip
        if p == g:
            tp += 1
            per_field[f] = (1, 0, 0)
        elif p[0] == "missing":
            fn += 1
            per_field[f] = (0, 0, 1)
        else:
            fp += 1
            fn += 1
            per_field[f] = (0, 1, 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "per_field": per_field,
            "n": tp + fp + fn}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return round(prec, 3), round(rec, 3), round(f1, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    pred = json.load(open(args.pred))
    gt = {inv["id"]: inv for inv in json.load(open(args.gt))["invoices"]}

    fields = ["vendor", "invoice_number", "invoice_date", "currency",
              "subtotal", "tax", "total"]
    agg: dict[str, list[int]] = {}  # field -> [tp, fp, fn] summed over docs
    n_docs = 0
    for doc_id, g in gt.items():
        p = pred.get(doc_id, {}).get("fields", {})
        s = field_scores(p, g.get("fields", {}), fields)
        n_docs += 1
        print(f"{doc_id}: P={s['precision']} R={s['recall']} F1={s['f1']}")
        for f, (a, b, c) in s["per_field"].items():
            t = agg.setdefault(f, [0, 0, 0])
            t[0] += a
            t[1] += b
            t[2] += c

    print(f"\nEvaluated {n_docs} documents.")
    if agg:
        # MICRO: pool every field prediction across the corpus.
        mtp = sum(t[0] for t in agg.values())
        mfp = sum(t[1] for t in agg.values())
        mfn = sum(t[2] for t in agg.values())
        mp, mr, mf = _prf(mtp, mfp, mfn)
        # MACRO: mean of per-field F1 (every field counts the same).
        f1s = [_prf(*t)[2] for t in agg.values()]
        macro_f1 = round(sum(f1s) / len(f1s), 3)
        print(f"Aggregate MICRO  P={mp} R={mr} F1={mf}  (n={mtp + mfp + mfn})")
        print(f"Aggregate MACRO  F1={macro_f1}  (mean over {len(agg)} fields)")
        print("Per-field: " +
              ", ".join(f"{f} F1={_prf(*t)[2]}" for f, t in sorted(agg.items())))

    # CER/WER needs jiwer + raw OCR text pairs — wire it here once you have them:
    #   from jiwer import cer, wer
    #   print("CER:", cer(gt_texts, pred_texts), "WER:", wer(gt_texts, pred_texts))


if __name__ == "__main__":
    main()
