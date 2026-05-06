#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pure Inner-GRPO ablation chain (stage 2/3):
# Train 16-sess with inner=0.0 from a pure 8-sess inner0 checkpoint.
#
# Usage:
#   srun --jobid=<job_id> --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_16sess_inner0_pure.sh
#
# After completion, use the produced global_step_5 hf_fixed path as
# CURRENT_MODEL_PATH_OVERRIDE in vllm_client_32sess_inner0_topk30_pure.sh.

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="${CURRENT_MODEL_PATH_OVERRIDE:-<repo>/checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_10/hf_fixed}"

export MAX_NUM_TURNS="${MAX_NUM_TURNS:-6}"
export INNER_GPRO_FRAC=0.0
export QA_TOP_K_PER_STAGE_OVERRIDE=50
export STAGES_OVERRIDE=16
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.2

export RUN_TAG="${RUN_TAG:-16sess_inner0_pure}"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"

