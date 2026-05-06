"""Re-judge a LoCoMo qa_dump with gpt-4o-mini, producing the same JSON shape
as results/judge_scores_gpt4omini/<run>_4omini.json.

Usage:
    python scripts/rejudge_locomo_dump.py \
        --dump_root qa_dumps/test_re_comp01_RETRY_7B_n1_20260429_235824/test/step_unknown \
        --out_path  results/judge_scores_gpt4omini/7b_lambda01_clean_v2_4omini.json \
        [--max_workers 8]
"""
import argparse, json, os, re, sys, time
import concurrent.futures
from collections import defaultdict, Counter
from statistics import mean
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('<repo>/.env')
client = OpenAI()

ACCURACY_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given:
(1) a question, (2) a gold (ground truth) answer, (3) a generated answer.

The gold answer is usually concise; the generated answer may be longer.
Be generous: if the generated answer touches on the same topic/date as
the gold, count CORRECT. Different formats for the same date (e.g. "May 7"
vs "7 May") are CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First give a one-sentence reasoning, then finish with CORRECT or WRONG. Do
NOT include both.

Return a JSON object with key "label" whose value is exactly "CORRECT"
or "WRONG"."""

CATEGORY_NAMES = {1: 'multi_hop', 2: 'temporal', 3: 'open_domain', 4: 'single_hop'}


def judge_one(rec, model='gpt-4o-mini'):
    msg = ACCURACY_PROMPT.format(
        question=rec['question'],
        gold_answer=rec['gold_answer'],
        generated_answer=rec['predicted_answer'],
    )
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': msg}],
                response_format={'type': 'json_object'},
                temperature=0,
                max_tokens=200,
            )
            content = r.choices[0].message.content
            obj = json.loads(content)
            label = (obj.get('label') or '').upper()
            return label, content
        except Exception as e:
            if attempt == 3:
                return 'ERROR', str(e)[:200]
            time.sleep(1.0 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_root', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--max_workers', type=int, default=8)
    ap.add_argument('--model', default='gpt-4o-mini')
    args = ap.parse_args()

    items = []
    for f in sorted(os.listdir(args.dump_root)):
        for line in open(os.path.join(args.dump_root, f)):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            items.append(r)
    print(f'Loaded {len(items)} records from {args.dump_root}', flush=True)

    judged = [None] * len(items)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(judge_one, items[i], args.model): i for i in range(len(items))}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            label, raw = fut.result()
            judged[i] = {'label': label, 'raw': raw}
            done += 1
            if done % 50 == 0:
                print(f'  {done}/{len(items)} ({100*done/len(items):.1f}%)  '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)

    # Aggregate
    enriched = []
    for it, j in zip(items, judged):
        cat_id = it.get('category')
        cat_name = it.get('category_name') or CATEGORY_NAMES.get(cat_id, str(cat_id))
        enriched.append({
            **it,
            'category_name': cat_name,
            'judge_label': j['label'],
            'judge_correct': 1 if j['label'] == 'CORRECT' else 0,
        })

    overall = {
        'n': len(enriched),
        'f1_mean': mean(x['f1'] for x in enriched),
        'bleu_mean': mean(x.get('bleu', 0) for x in enriched),
        'llm_judge_mean': mean(x['judge_correct'] for x in enriched),
        'label_dist': dict(Counter(x['judge_label'] for x in enriched)),
    }
    per_cat = defaultdict(list)
    for x in enriched:
        per_cat[x['category_name']].append(x)
    per_cat_summary = {}
    for cn, lst in per_cat.items():
        per_cat_summary[cn] = {
            'n': len(lst),
            'f1_mean': mean(x['f1'] for x in lst),
            'bleu_mean': mean(x.get('bleu', 0) for x in lst),
            'llm_judge_mean': mean(x['judge_correct'] for x in lst),
        }

    out = {
        'summary': {
            'dump_root': args.dump_root,
            'judge_model': args.model,
            'base_url': 'https://api.openai.com/v1',
            'num_items': len(enriched),
            'overall': overall,
            'per_category': per_cat_summary,
        },
        'items': enriched,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n=== {args.out_path} ===')
    print(f'F1 = {100*overall["f1_mean"]:.2f}  B1 = {100*overall["bleu_mean"]:.2f}  '
          f'J = {100*overall["llm_judge_mean"]:.2f}  (n={len(enriched)})')
    for cn, s in per_cat_summary.items():
        print(f"  {cn:>14s}: n={s['n']:>4d}  F1={100*s['f1_mean']:.2f}  "
              f"B1={100*s['bleu_mean']:.2f}  J={100*s['llm_judge_mean']:.2f}")


if __name__ == '__main__':
    main()
