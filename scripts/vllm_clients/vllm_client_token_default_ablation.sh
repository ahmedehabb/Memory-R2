#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CLEAN clip-mode ablation: BOTH clip_mode=token AND agg_mode=token (standard PPO defaults).
# Compares against the paper contribution of clip_mode=turn + agg_mode=turn (turn-level ratio + turn-level loss).
# Earlier runs only flipped one knob (token-clip + turn-agg, OR token-clip + traj-agg) — neither is the
# clean "no length weighting" baseline. This script flips both, matching the verl config defaults.
#
# Required env (one of):
#   STAGES_OVERRIDE — typically 8 (matched to existing 8sess_clip01_comp02_thresh05 turn baseline)
#                     or 32 (matched to the 7B 32sess(λ=0.3) curriculum sweet spot for tab:ablation)
# Optional env:
#   COMP_LAMBDA (default 0.2 to match 8sess_clip01_comp02_thresh05 baseline; use 0.3 for 32-sess match)
#   WARM_START_PATH (required when STAGES_OVERRIDE=32; should be 7B 8sess(λ=COMP_LAMBDA) hf_fixed)
#   RUN_TAG (default: token_default_<sess>sess_lambda<L>)

export SKIP_NODE_CHECK=1

: "${STAGES_OVERRIDE:?Set STAGES_OVERRIDE=8 or 32}"
: "${COMP_LAMBDA:=0.2}"

TAG_LAMBDA="${COMP_LAMBDA//./}"
: "${RUN_TAG:=token_default_${STAGES_OVERRIDE}sess_lambda${TAG_LAMBDA}}"
export RUN_TAG

# Flip BOTH ratio-clip and loss-aggregation to token-level (standard PPO defaults).
export CLIP_MODE=token
export AGG_MODE=token

# Match the existing turn-level baselines on every other axis so the ablation isolates
# clip_mode/agg_mode only.
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export INNER_GPRO_FRAC=0.5
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export EPOCHS_PER_STAGE_OVERRIDE=10
export COMPRESSION_PENALTY=$COMP_LAMBDA

# 32-sess variant must warm-start from a matched 8sess(λ) checkpoint to mirror the curriculum.
if [ "$STAGES_OVERRIDE" = "32" ]; then
    : "${WARM_START_PATH:?For STAGES_OVERRIDE=32 set WARM_START_PATH to 7B 8sess(λ=$COMP_LAMBDA) hf_fixed}"
    export CURRENT_MODEL_PATH_OVERRIDE="$WARM_START_PATH"
    export EPOCHS_PER_STAGE_OVERRIDE=5
fi

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
