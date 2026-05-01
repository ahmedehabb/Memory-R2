#!/usr/bin/env bash
# Smoke-test: conv-43 only, 2 sessions, gpt-4o.
# Usage: bash sft/test_collect_traces.sh  (from repo root)
#    or: bash test_collect_traces.sh      (from sft/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"

conda run -n rema python "$SCRIPT_DIR/collect_answer_traces.py" \
    --model        gpt-4o \
    --conv_ids     conv-43 \
    --max_sessions 2 \
    --output       "$REPO_ROOT/data/sft/test_traces.jsonl" \
    --verbose

echo ""
echo "=== Output preview ==="
python3 -c "
import json, re
with open('$REPO_ROOT/data/sft/test_traces.jsonl') as f:
    records = [json.loads(l) for l in f]
for i, r in enumerate(records):
    meta = r['metadata']
    m = re.search(r'<answer>(.*?)</answer>', r['messages'][1]['content'], re.DOTALL|re.IGNORECASE)
    ans = m.group(1).strip() if m else '[NO TAG]'
    f1 = meta['f1_score']
    bar = '█' * int(f1*10) + '░' * (10-int(f1*10))
    print(f'[{i}] F1={f1:.2f} |{bar}|  GT: {meta[\"ground_truth\"]!r:40s}  A: {ans!r}')
avg = sum(r['metadata']['f1_score'] for r in records) / len(records)
print(f'\nMean F1: {avg:.3f}')
"
