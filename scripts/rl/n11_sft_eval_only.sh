#!/usr/bin/env bash
# Eval-only: connect to an SFT-answer server already running on a DIFFERENT
# allocation and run the LoCoMo test eval against the 32sess_champion_v2 mem
# policy. Pairs with n11_sft_serve_only.sh (which writes vllm_servers_qwen_<LABEL>/).
#
# Required env: SFT_LABEL (must match the server's SFT_LABEL)
# Optional env: MEM_CKPT_PATH, EVAL_GPU_MEM_UTIL (default 0.85 since this node is eval-only)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SFT_LABEL="${SFT_LABEL:?SFT_LABEL=<short label, must match server> required}"
TS="$(date +%Y%m%d_%H%M%S)"
ISO_DIR="$REPO_ROOT/vllm_servers_qwen_${SFT_LABEL}"

# Wait up to 20 min for the paired server to register
for i in $(seq 1 240); do
  if ls "$ISO_DIR"/server_*.txt >/dev/null 2>&1; then
    URL=$(cat "$ISO_DIR"/server_0.txt 2>/dev/null)
    if [ -n "$URL" ] && curl -fsS "http://${URL}/v1/models" > /dev/null 2>&1; then
      echo "[n11_eval_only] paired server ready at $URL after $((i*5))s"
      break
    fi
  fi
  sleep 5
done

if ! ls "$ISO_DIR"/server_*.txt >/dev/null 2>&1; then
  echo "[n11_eval_only] ERROR: server never registered in $ISO_DIR"
  exit 1
fi

export JUDGE_RENDEZVOUS_DIR="$ISO_DIR"
export MODEL_PATH_OVERRIDE="${MEM_CKPT_PATH:-$REPO_ROOT/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed}"
export RUN_TAG="${SFT_LABEL}_eval_${TS}"
export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.85}"

echo "[n11_eval_only] starting test-eval RUN_TAG=$RUN_TAG (eval_mem=$EVAL_GPU_MEM_UTIL)"
exec bash "$REPO_ROOT/scripts/vllm_clients/vllm_client_test_eval_qwen.sh"
