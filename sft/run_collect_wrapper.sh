#!/bin/bash
set -euo pipefail

MODEL_NAME="${1:-gpt-4o}"
TS="$(date +%Y%m%d_%H%M%S)"
MODEL_SAFE="$(echo "$MODEL_NAME" | tr -cs '[:alnum:]._+-' '_' )"
OUTPUT_PATH="data/sft/answer_traces_${MODEL_SAFE}_${TS}.jsonl"
MAX_TURNS_PER_SESSION="${MAX_TURNS_PER_SESSION:-1}"
TOP_K_MEMORIES="${TOP_K_MEMORIES:-20}"
TOP_K_QA_PER_SPEAKER="${TOP_K_QA_PER_SPEAKER:-30}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.1}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[run_collect_wrapper] ERROR: OPENAI_API_KEY is not set."
    echo "Set it first, e.g. export OPENAI_API_KEY='sk-...'"
    exit 1
fi

cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

echo "[run_collect_wrapper] Model: $MODEL_NAME"
echo "[run_collect_wrapper] Output: $OUTPUT_PATH"
echo "[run_collect_wrapper] max_turns_per_session: $MAX_TURNS_PER_SESSION"
echo "[run_collect_wrapper] top_k_memories: $TOP_K_MEMORIES"
echo "[run_collect_wrapper] top_k_qa_per_speaker: $TOP_K_QA_PER_SPEAKER"
echo "[run_collect_wrapper] similarity_threshold: $SIMILARITY_THRESHOLD"

conda run --no-capture-output -n rema python -u sft/collect_answer_traces.py \
        --model "$MODEL_NAME" \
        --conv_ids conv-43 conv-47 \
    --max_turns_per_session "$MAX_TURNS_PER_SESSION" \
    --top_k_memories "$TOP_K_MEMORIES" \
    --top_k_qa_per_speaker "$TOP_K_QA_PER_SPEAKER" \
    --similarity_threshold "$SIMILARITY_THRESHOLD" \
        --output "$OUTPUT_PATH" \
        --verbose

echo "[run_collect_wrapper] Done: $OUTPUT_PATH"
