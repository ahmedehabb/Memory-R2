#!/usr/bin/env python
"""Standalone LLM-judge scorer for LoCoMo QA dumps emitted by rema.py.

The reward manager (`src/verl/verl/workers/reward_manager/rema.py`) writes one
JSONL per (conv, chunk, epoch, index_in_batch) under the layout:

  $REMA_QA_DUMP_DIR/<run>/<split>/step_<step>/conv<c>_chunk<k>_epoch<e>_idx<i>.jsonl

Each line is one QA with {question, gold_answer, predicted_answer, response,
category, evidence, f1, bleu, ...}. We don't recompute f1/bleu (rema.py already
did) — we only call the OpenAI-compatible LLM-judge to produce a binary
CORRECT/WRONG label per QA, then aggregate overall + per-category means.

Env:
  OPENAI_BASE_URL      (e.g. http://hkn1958.localdomain:8107/v1 for local gpt-oss-120b)
  OPENAI_API_KEY       (EMPTY for local vLLM; real key for GPT-4o)
  OPENAI_JUDGE_MODEL   (e.g. gpt-4o or openai/gpt-oss-120b; default gpt-4o-mini)

Usage:
  OPENAI_BASE_URL=http://hkn1958.localdomain:8107/v1 \
  OPENAI_API_KEY=EMPTY \
  OPENAI_JUDGE_MODEL=openai/gpt-oss-120b \
  python testing/pipeline_test_locomo_qa_dump/score_locomo_qa_dumps.py \
    --dump_root qa_dumps/<run_name>/test/step_<step> \
    --output results/scored_locomo_<run>_step<step>_gptoss.json \
    --max_workers 16
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict

from openai import OpenAI


_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question, (2) a gold (ground truth) answer, (3) a generated answer.

The gold answer is usually concise; the generated answer may be longer. Be generous: if the generated answer touches on the same topic/date as the gold, count CORRECT. Different formats for the same date (e.g. "May 7" vs "7 May") are CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First give a one-sentence reasoning, then finish with CORRECT or WRONG. Do NOT include both.
Return a JSON object with key "label" whose value is exactly "CORRECT" or "WRONG"."""


# LoCoMo numeric categories. Must match rema.py::CATEGORY_NAMES exactly.
# Category 5 is adversarial/unanswerable — we skip it during scoring.
LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial_skip",
}


def extract_label(text):
    try:
        m = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).upper().strip()
            if lab in ("CORRECT", "WRONG"):
                return lab
    except Exception:
        pass
    matches = re.findall(r"\b(CORRECT|WRONG)\b", text.upper())
    if matches:
        return matches[-1]
    return None


def judge_once(question, gold, pred, max_retries=3):
    prompt = ACCURACY_PROMPT.format(question=question, gold_answer=gold, generated_answer=pred)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
            lab = extract_label(raw)
            if lab == "CORRECT":
                return 1, raw
            if lab == "WRONG":
                return 0, raw
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"[judge error] {e}"
            time.sleep(1.5 * (attempt + 1))
    return None, "[judge error] no valid label after retries"


def process_item(record):
    cat = record.get("category", 0)
    try:
        cat_int = int(cat)
    except Exception:
        cat_int = 0
    if cat_int == 5:
        return None
    q = str(record.get("question", ""))
    gold = str(record.get("gold_answer", ""))
    pred = str(record.get("predicted_answer", ""))
    ll, judge_raw = judge_once(q, gold, pred)
    return {
        "conv_id": record.get("conv_id"),
        "chunk_id": record.get("chunk_id"),
        "qa_idx": record.get("qa_idx"),
        "category": cat_int,
        "category_name": LOCOMO_CATEGORY_NAMES.get(cat_int, "unknown"),
        "question": q,
        "gold_answer": gold,
        "predicted_answer": pred,
        "f1": record.get("f1"),
        "bleu": record.get("bleu"),
        "llm_score": ll,
        "llm_raw": judge_raw,
    }


def load_dumps(dump_root):
    files = sorted(glob.glob(os.path.join(dump_root, "**", "*.jsonl"), recursive=True))
    records = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", required=True,
                    help="Directory containing conv*.jsonl files (recursively searched).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    print(f"[score] judge={_JUDGE_MODEL} base_url={_BASE_URL} dump_root={args.dump_root}")
    records = load_dumps(args.dump_root)
    print(f"[score] loaded {len(records)} QA records")
    if args.limit:
        records = records[: args.limit]

    results = []
    results_lock = threading.Lock()
    done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process_item, r) for r in records]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            done += 1
            if r is None:
                continue
            with results_lock:
                results.append(r)
                if done % 50 == 0 or done == len(records):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-3)
                    print(f"[score] {done}/{len(records)}  rate={rate:.1f}/s")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category_name"]].append(r)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "dump_root": args.dump_root,
        "judge_model": _JUDGE_MODEL,
        "base_url": _BASE_URL,
        "num_items": len(results),
        "overall": {
            "bleu_mean": mean([r["bleu"] for r in results]),
            "f1_mean": mean([r["f1"] for r in results]),
            "llm_judge_mean": mean([r["llm_score"] for r in results]),
        },
        "per_category": {
            cat: {
                "n": len(items),
                "bleu_mean": mean([x["bleu"] for x in items]),
                "f1_mean": mean([x["f1"] for x in items]),
                "llm_judge_mean": mean([x["llm_score"] for x in items]),
            }
            for cat, items in by_cat.items()
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[score] wrote {args.output}")


if __name__ == "__main__":
    main()
