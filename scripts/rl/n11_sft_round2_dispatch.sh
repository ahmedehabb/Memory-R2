#!/usr/bin/env bash
# Round 2 of paired SFT-answer evals (step30 + step40).
# Auto-runs after Round 1 finishes. Uses the same node pairing:
#   Pair A: 3982259 (H200x4) hosts step30 server, 3985704 (H100x4) runs eval
#   Pair B: 3985761 (H100x4) hosts step40 server, 3986106 (H100x4) runs eval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

REPO=$REPO_ROOT
TS=$(date +%H%M%S)

# Pair A: step30
srun --jobid=3982259 --overlap -N1 -n1 \
  env SFT_HF_PATH=$REPO/outputs/answer_agent_sft_hf/n11_sft_step30_20260426_130752 \
      SFT_LABEL=n11_sft_step30_paired \
      VLLM_SERVER_GPU_MEM_UTIL=0.85 \
  bash scripts/rl/n11_sft_serve_only.sh \
  > logs/3982259_n11_sft_step30_serve_only_${TS}.log 2>&1 &
echo "step30 SERVER PID=$! on 3982259"

srun --jobid=3985704 --overlap -N1 -n1 \
  env SFT_LABEL=n11_sft_step30_paired \
      EVAL_GPU_MEM_UTIL=0.85 \
  bash scripts/rl/n11_sft_eval_only.sh \
  > logs/3985704_n11_sft_step30_eval_only_${TS}.log 2>&1 &
echo "step30 EVAL PID=$! on 3985704"

# Pair B: step40
srun --jobid=3985761 --overlap -N1 -n1 \
  env SFT_HF_PATH=$REPO/outputs/answer_agent_sft_hf/n11_sft_step40_20260426_130752 \
      SFT_LABEL=n11_sft_step40_paired \
      VLLM_SERVER_GPU_MEM_UTIL=0.85 \
  bash scripts/rl/n11_sft_serve_only.sh \
  > logs/3985761_n11_sft_step40_serve_only_${TS}.log 2>&1 &
echo "step40 SERVER PID=$! on 3985761"

srun --jobid=3986106 --overlap -N1 -n1 \
  env SFT_LABEL=n11_sft_step40_paired \
      EVAL_GPU_MEM_UTIL=0.85 \
  bash scripts/rl/n11_sft_eval_only.sh \
  > logs/3986106_n11_sft_step40_eval_only_${TS}.log 2>&1 &
echo "step40 EVAL PID=$! on 3986106"

echo "Round 2 dispatched at TS=$TS"
