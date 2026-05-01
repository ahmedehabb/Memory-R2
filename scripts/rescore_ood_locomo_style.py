#!/usr/bin/env python
"""Re-score OOD evals (LongMemEval / MSC / MemBench) with LoCoMo-style F1/BLEU
to make tab:ood-datasets numerically comparable to tab:main / tab:compression.

LoCoMo scorer (score_locomo_qa_dumps.py) uses:
  - F1: multiset Counter intersection / multiset pred & gold lengths, normalize_text
  - BLEU-1: pure precision (Counter clipped / |pred|), no brevity penalty
  - normalize_text: lowercase, strip punctuation, remove articles, collapse whitespace

Usage:
  python rescore_ood_locomo_style.py <input.json> [--out <out.json>]

Input JSON formats supported:
  1. {"summary": {...}, "items": [{"question","answer","response_extracted","llm_score",...}, ...]}
  2. {"5": [{...}, "9": [{...}], ...} (evals.py style — categories or speakers as keys)
  3. [{"question","answer","response","llm_score",...}, ...]
"""
import argparse, json, re, sys
from collections import Counter
from statistics import mean


_ART_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(s):
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ART_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def f1_locomo(pred, gold):
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    P = num_same / len(p)
    R = num_same / len(g)
    return 2 * P * R / (P + R)


def bleu1_locomo(pred, gold):
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    return overlap / max(len(p), 1)


def extract_pred(item):
    """Best-effort extraction of predicted-answer text from various item formats."""
    for k in ("response_extracted", "predicted_answer", "response", "answer_pred", "model_answer"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def gather_items(data):
    """Yield items irrespective of top-level structure."""
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                yield it
            return
        # Treat dict-of-lists (categories or speaker keys) as flattened
        flat = []
        for k, v in data.items():
            if isinstance(v, list):
                flat.extend(v)
        if flat:
            for it in flat:
                yield it
            return
    if isinstance(data, list):
        for it in data:
            yield it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)

    f1_scores, bleu_scores, llm_scores = [], [], []
    by_cat = {}
    n = 0
    for item in gather_items(d):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", ""))
        a = str(item.get("answer", "") or item.get("gold_answer", ""))
        p = extract_pred(item)
        if not q and not a:
            continue
        f1 = f1_locomo(p, a)
        b1 = bleu1_locomo(p, a)
        llm = item.get("llm_score")
        if isinstance(llm, str):
            llm = 1.0 if llm.upper() in ("CORRECT", "TRUE", "YES", "1") else 0.0
        elif llm is None:
            llm = item.get("llm_judge")
        try:
            llm = float(llm) if llm is not None else None
        except Exception:
            llm = None

        f1_scores.append(f1)
        bleu_scores.append(b1)
        if llm is not None:
            llm_scores.append(llm)

        cat = str(item.get("category", item.get("question_type", "all")))
        by_cat.setdefault(cat, {"f1": [], "bleu": [], "llm": []})
        by_cat[cat]["f1"].append(f1)
        by_cat[cat]["bleu"].append(b1)
        if llm is not None:
            by_cat[cat]["llm"].append(llm)
        n += 1

    overall = {
        "n": n,
        "f1_mean_locomo": mean(f1_scores) if f1_scores else 0.0,
        "bleu_mean_locomo": mean(bleu_scores) if bleu_scores else 0.0,
        "llm_mean": mean(llm_scores) if llm_scores else None,
    }
    cat_summary = {}
    for cat, vals in by_cat.items():
        cat_summary[cat] = {
            "n": len(vals["f1"]),
            "f1_mean_locomo": mean(vals["f1"]) if vals["f1"] else 0.0,
            "bleu_mean_locomo": mean(vals["bleu"]) if vals["bleu"] else 0.0,
            "llm_mean": mean(vals["llm"]) if vals["llm"] else None,
        }

    out = {
        "input": args.input,
        "scorer": "score_locomo_qa_dumps-style (multiset F1, no-BP BLEU-1, normalize_text article-stripped)",
        "overall": overall,
        "by_category": cat_summary,
    }

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
