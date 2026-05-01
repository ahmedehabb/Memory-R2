#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess continuation from comp03 step5 — inner_n=8 + topk_memories=50
#
# Goal: push beyond 0.501 by (a) doubling inner rollouts (4→8, matching 16-sess
#       inner_n8 which gave +0.021 gap vs inner_n4) and (b) increasing memory
#       retrieval budget (25→50 candidates per query, from 700+ stored items).
#
# Start: 32sess_fixedqa_comp03 step5 (val=0.491, mfail=0.067, test=0.498)
# Same fixedQA reward config as comp03 (topk_qa=60, inner=32, terminal=128).
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_cont_inner8_topk50.sh > logs/<job_id>/32sess_cont_inner8_topk50_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_fixedqa_comp03__20260402_233703_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# Fixed QA coverage (same as comp03)
export QA_TOP_K_PER_STAGE_OVERRIDE=60
export REMA_REWARD_MAX_QA_TRAIN_INNER=32
export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=128

export COMPRESSION_PENALTY=0.3
export INNER_GPRO_FRAC=0.5

# KEY CHANGES vs comp03:
export INNER_N=8           # was 4 — doubles inner rollouts (16-sess inner8 gave +0.021)
export TOP_K_MEMORIES=50   # was 25 (default) — more memory candidates per query at 32-sess

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_cont_inner8_topk50"

export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
