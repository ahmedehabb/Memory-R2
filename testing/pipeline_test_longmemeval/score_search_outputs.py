#!/usr/bin/env python
"""
Standalone scorer for rema_search output JSONs.

Reads a results JSON of the shape:
  {
    "<conv_or_question_key>": [
      {"question": ..., "answer": <gold>, "response": <pred>, "category": ...},
      ...
    ],
    ...
  }

For each item (skipping category == "5"), computes:
  - BLEU-1 via sacrebleu (unigram)
  - Token-F1 (standard squad-style)
  - LLM-judge binary CORRECT/WRONG via OpenAI-compatible client

Respects env vars so we can route through the local vLLM gpt-oss-120b judge:
  OPENAI_BASE_URL   (e.g. http://hkn0922.localdomain:8107/v1)
  OPENAI_API_KEY    (EMPTY for local vLLM)
  OPENAI_JUDGE_MODEL (default: gpt-4o-mini; for us: openai/gpt-oss-120b)

Writes:
  <output_file>  : per-item JSON with f1/bleu/llm_score
  stdout         : aggregate mean scores + per-category

Usage:
  OPENAI_BASE_URL=http://hkn0922.localdomain:8107/v1 \
  OPENAI_API_KEY=EMPTY \
  OPENAI_JUDGE_MODEL=openai/gpt-oss-120b \
  python score_search_outputs.py \
    --input results/longmemeval_32sess_results/longmemeval_rema_qwen_32sess_top30.json \
    --output results/scored_longmemeval_32sess_gptoss.json \
    --max_workers 16
"""
import argparse
import concurrent.futures
import json
import os
import re
import string
import sys
import threading
import time
from collections import Counter, defaultdict

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


def _normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s).strip()
    return s


def token_f1(pred, gold):
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu1(pred, gold):
    p_tokens = _normalize(pred).split()
    g_tokens = _normalize(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0
    p_counts = Counter(p_tokens)
    g_counts = Counter(g_tokens)
    overlap = sum((p_counts & g_counts).values())
    return overlap / max(len(p_tokens), 1)


def extract_answer_from_text(text: str) -> str:
    """Mirror of verl/rema_trainer/memory/utils/parse_response.extract_answer_from_text.

    Matches the LoCoMo answer-agent pipeline's own extraction so our scoring
    here is apples-to-apples with how LoCoMo F1/BLEU-1 are computed upstream.
    """
    try:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"\*\*Answer:\*\*\s*(.*)", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"\*\*Answer:\s*(.*?)\*\*", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"Answer:\s*(.*)", text)
        if m:
            return m.group(1).strip()
        return text.strip()
    except Exception:
        return text.strip()


def extract_label(text):
    # Primary: try JSON
    try:
        m = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).upper().strip()
            if lab in ("CORRECT", "WRONG"):
                return lab
    except Exception:
        pass
    # Fallback: last CORRECT/WRONG token in text
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


def process_item(conv_key, item, idx):
    q = str(item.get("question", ""))
    gold = str(item.get("answer", ""))
    raw_pred = str(item.get("response", ""))
    cat = str(item.get("category", ""))
    if cat == "5":
        return None
    # Extract final answer from the verbose response, matching the LoCoMo pipeline.
    extracted_pred = extract_answer_from_text(raw_pred)
    f1 = token_f1(extracted_pred, gold)
    b1 = bleu1(extracted_pred, gold)
    ll, judge_raw = judge_once(q, gold, extracted_pred)
    return {
        "conv_key": conv_key,
        "idx": idx,
        "question": q,
        "answer": gold,
        "response_raw": raw_pred,
        "response_extracted": extracted_pred,
        "category": cat,
        "bleu_score": b1,
        "f1_score": f1,
        "llm_score": ll,
        "llm_raw": judge_raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    print(f"[score] judge={_JUDGE_MODEL} base_url={_BASE_URL} input={args.input}")

    with open(args.input) as f:
        data = json.load(f)

    tasks = []
    for conv_key, lst in data.items():
        for i, item in enumerate(lst):
            tasks.append((conv_key, item, i))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[score] total items: {len(tasks)}")

    results = []
    results_lock = threading.Lock()
    done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process_item, ck, it, i) for ck, it, i in tasks]
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            done += 1
            if r is None:
                continue
            with results_lock:
                results.append(r)
                if done % 50 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-3)
                    print(f"[score] {done}/{len(tasks)}  rate={rate:.1f}/s")

    # Aggregate
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "input": args.input,
        "judge_model": _JUDGE_MODEL,
        "base_url": _BASE_URL,
        "num_items": len(results),
        "overall": {
            "bleu_score_mean": mean([r["bleu_score"] for r in results]),
            "f1_score_mean": mean([r["f1_score"] for r in results]),
            "llm_score_mean": mean([r["llm_score"] for r in results]),
        },
        "per_category": {
            cat: {
                "n": len(items),
                "bleu_score_mean": mean([x["bleu_score"] for x in items]),
                "f1_score_mean": mean([x["f1_score"] for x in items]),
                "llm_score_mean": mean([x["llm_score"] for x in items]),
            }
            for cat, items in by_cat.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "items": results}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[score] wrote {args.output}")


if __name__ == "__main__":
    main()
