#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 7B 8-sess λ-CONSISTENT training (Plan B stage 1).
# Warm-starts from Qwen2.5-7B-Instruct base, trains 8sess at target λ.
#
# Required env:
#   COMP_LAMBDA — target compression coefficient (e.g. 0.0, 0.05, 0.3, 0.5)
# Optional env:
#   RUN_TAG (default: p7_8sess_lambda<L>_consistent)

export SKIP_NODE_CHECK=1

if [ -z "$COMP_LAMBDA" ]; then
    echo "[7b-8sess-sweep] ERROR: COMP_LAMBDA required (e.g. 0.0, 0.05, 0.3, 0.5)"
    exit 1
fi

TAG_LAMBDA="${COMP_LAMBDA//./}"
: "${RUN_TAG:=p7_8sess_lambda${TAG_LAMBDA}_consistent}"
export RUN_TAG

# Default to Qwen/Qwen2.5-7B-Instruct via standalone defaults (no override needed)

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=8
export EPOCHS_PER_STAGE_OVERRIDE=10
export COMPRESSION_PENALTY=$COMP_LAMBDA

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
