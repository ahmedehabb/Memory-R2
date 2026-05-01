#!/usr/bin/env bash
# Full SFT data collection: both training convs, all sessions, gpt-4o.
# Usage: bash sft/run_full.sh  (from repo root)
#    or: bash run_full.sh      (from sft/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"

conda run -n rema python "$SCRIPT_DIR/collect_answer_traces.py" \
    --model          gpt-4o \
    --conv_ids       conv-43 conv-47 \
    --output         "$REPO_ROOT/data/sft/answer_traces.jsonl" \
    --verbose

echo ""
echo "=== Summary ==="
python3 -c "
import json
with open('$REPO_ROOT/data/sft/answer_traces.jsonl') as f:
    records = [json.loads(l) for l in f]
avg = sum(r['metadata']['f1_score'] for r in records) / len(records)
by_conv = {}
for r in records:
    c = r['metadata']['conv_id']
    by_conv.setdefault(c, []).append(r['metadata']['f1_score'])
print(f'Total QA samples : {len(records)}')
print(f'Overall mean F1  : {avg:.3f}')
for c, scores in sorted(by_conv.items()):
    print(f'  {c}: {len(scores)} samples, mean F1={sum(scores)/len(scores):.3f}')
"
