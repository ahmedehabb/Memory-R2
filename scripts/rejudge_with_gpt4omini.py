#!/usr/bin/env python
"""Re-judge an OOD JSON's items with OpenAI gpt-4o-mini (replacing gpt-oss-120b scoring).

Reads an OOD scoring JSON (with `items[*].{question, answer, response_extracted}`)
or `lme_s_cleaned_*` style. For each item, calls OpenAI gpt-4o-mini with the same
accuracy prompt the OOD pipeline uses, parses CORRECT/WRONG, aggregates llm_judge mean.

Output: writes a new JSON with overall {f1_mean (LoCoMo-style), bleu_mean (LoCoMo-style),
llm_judge_mean (gpt-4o-mini)} that supersedes both the old gpt-oss llm_score and the
OOD-pipeline F1/BLEU.
"""
import argparse, concurrent.futures, json, os, re, sys, time
from collections import Counter
from statistics import mean
from dotenv import load_dotenv
from openai import OpenAI


# Use the same accuracy prompt the OOD pipeline uses (lme/msc/membench all share it).
ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First provide a short reasoning sentence and then output a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}."""


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
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0: return 0.0
    P = n/len(p); R = n/len(g)
    return 2*P*R/(P+R)


def bleu1_locomo(pred, gold):
    p = normalize_text(pred).split(); g = normalize_text(gold).split()
    if not p or not g: return 0.0
    return sum((Counter(p)&Counter(g)).values())/max(len(p),1)


def extract_pred(item):
    for k in ("response_extracted","predicted_answer","response","answer_pred","model_answer"):
        v = item.get(k)
        if v is not None and str(v).strip(): return str(v)
    return ""


def gather_items(d):
    if isinstance(d, dict):
        if "items" in d and isinstance(d["items"], list):
            for it in d["items"]: yield it
            return
        flat=[]
        for k,v in d.items():
            if isinstance(v,list): flat.extend(v)
        if flat:
            for it in flat: yield it
            return
    if isinstance(d,list):
        for it in d: yield it


def parse_label(text):
    try:
        m = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label","")).upper().strip()
            if lab in ("CORRECT","WRONG"): return lab
    except Exception: pass
    # fallback
    matches = re.findall(r"\b(CORRECT|WRONG)\b", text.upper())
    if matches: return matches[-1]
    return None


def judge_one(client, model, q, a, p):
    prompt = ACCURACY_PROMPT.format(question=q, gold_answer=a, generated_answer=p)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=128,
            )
            return parse_label(r.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5*(attempt+1))
            else:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--env", default="testing/pipeline_test_longmemeval/.env")
    args = ap.parse_args()

    load_dotenv(args.env)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print(f"[err] no OPENAI_API_KEY in {args.env}", file=sys.stderr); sys.exit(1)
    client = OpenAI(api_key=key)

    with open(args.input) as f: d = json.load(f)
    items = list(gather_items(d))
    items = [it for it in items if isinstance(it, dict)]
    print(f"[rejudge] {len(items)} items in {args.input}")

    results = []
    n_correct = 0; n_done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_idx = {}
        for i, it in enumerate(items):
            q = str(it.get("question",""))
            a = str(it.get("answer","") or it.get("gold_answer",""))
            p = extract_pred(it)
            if not q or not a:
                results.append({"f1":0.0,"bleu":0.0,"llm":None})
                continue
            future_to_idx[ex.submit(judge_one, client, args.model, q, a, p)] = (i, q, a, p)

        for fut in concurrent.futures.as_completed(future_to_idx):
            i, q, a, p = future_to_idx[fut]
            label = fut.result()
            if label == "CORRECT":
                llm = 1.0; n_correct += 1
            elif label == "WRONG":
                llm = 0.0
            else:
                llm = None
            f1v = f1_locomo(p, a); b1v = bleu1_locomo(p, a)
            # store at right index
            while len(results) <= i: results.append(None)
            results[i] = {"f1":f1v,"bleu":b1v,"llm":llm,"q":q[:80],"a":a[:60],"p":p[:60]}
            n_done += 1
            if n_done % 50 == 0:
                rate = n_done/(time.time()-t0+1e-9)
                print(f"[rejudge] {n_done}/{len(future_to_idx)} rate={rate:.1f}/s n_correct={n_correct}", file=sys.stderr, flush=True)

    f1s = [r["f1"] for r in results if r and r.get("f1") is not None]
    b1s = [r["bleu"] for r in results if r and r.get("bleu") is not None]
    llms = [r["llm"] for r in results if r and r.get("llm") is not None]
    out = {
        "input": args.input,
        "judge_model": args.model,
        "scorer_f1_b1": "score_locomo_qa_dumps style (multiset, no-BP BLEU-1, article-stripped)",
        "n_items": len(results),
        "n_judge_valid": len(llms),
        "overall": {
            "f1_mean": mean(f1s) if f1s else 0.0,
            "bleu_mean": mean(b1s) if b1s else 0.0,
            "llm_judge_mean": mean(llms) if llms else 0.0,
        },
    }
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f: json.dump({"summary": out, "items": results}, f, indent=2)
    print(f"[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
