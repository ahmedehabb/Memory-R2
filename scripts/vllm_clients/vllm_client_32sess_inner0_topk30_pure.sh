#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pure Inner-GRPO ablation chain (stage 3/3):
# Train 32-sess with inner=0.0 and topk=30 from a pure 16-sess inner0 checkpoint.
#
# Required:
#   export CURRENT_MODEL_PATH_OVERRIDE=<path-to-16sess_inner0_pure/global_step_5/hf_fixed>
#
# Usage:
#   srun --jobid=<job_id> --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_32sess_inner0_topk30_pure.sh

set -euo pipefail

if [ -z "${CURRENT_MODEL_PATH_OVERRIDE:-}" ]; then
  echo "[pure-inner0-32] ERROR: CURRENT_MODEL_PATH_OVERRIDE is required."
  echo "[pure-inner0-32] Example:"
  echo "  export CURRENT_MODEL_PATH_OVERRIDE=/.../curr_16sess_16sess_inner0_pure.../global_step_5/hf_fixed"
  exit 1
fi

export SKIP_NODE_CHECK=1
export MAX_NUM_TURNS="${MAX_NUM_TURNS:-6}"
export INNER_GPRO_FRAC=0.0
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

export RUN_TAG="${RUN_TAG:-32sess_inner0_topk30_pure}"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"

