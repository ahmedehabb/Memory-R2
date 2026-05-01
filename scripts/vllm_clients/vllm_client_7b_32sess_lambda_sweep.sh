#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 7B 32-sess λ-CONSISTENT training (Plan B stage 2).
# Warm-starts from 7B 8sess(λ) ckpt matching the target λ.
#
# Required env:
#   COMP_LAMBDA — target compression coefficient (e.g. 0.0, 0.05, 0.1, 0.3, 0.5)
#   WARM_START_PATH — path to 7B 8sess(λ) hf_fixed (must match COMP_LAMBDA)
# Optional env:
#   RUN_TAG (default: p7_32sess_lambda<L>_consistent)

export SKIP_NODE_CHECK=1

if [ -z "$COMP_LAMBDA" ]; then
    echo "[7b-32sess-sweep] ERROR: COMP_LAMBDA required"
    exit 1
fi
if [ -z "$WARM_START_PATH" ]; then
    echo "[7b-32sess-sweep] ERROR: WARM_START_PATH required (7B 8sess(λ=$COMP_LAMBDA) hf_fixed)"
    exit 1
fi

TAG_LAMBDA="${COMP_LAMBDA//./}"
: "${RUN_TAG:=p7_32sess_lambda${TAG_LAMBDA}_consistent}"
export RUN_TAG

# λ-consistent warm-start: 7B 8sess(λ) → 7B 32sess(λ)
export CURRENT_MODEL_PATH_OVERRIDE="$WARM_START_PATH"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=$COMP_LAMBDA

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
