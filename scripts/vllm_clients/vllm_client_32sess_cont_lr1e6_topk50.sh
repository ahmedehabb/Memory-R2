#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess continuation from comp03 step5 — lower LR=1e-6 + topk_memories=50
#
# Goal: fine-tune the comp03 model more gently (LR halved 2e-6→1e-6) while also
#       expanding memory retrieval budget (25→50 candidates per query).
#       Rationale: comp03 val=0.491 at step5 vs step4=0.531 train/acc suggests
#       the last step may have been slightly too aggressive. Lower LR + more
#       memory context may yield more stable continued improvement.
#
# Start: 32sess_fixedqa_comp03 step5 (val=0.491, mfail=0.067, test=0.498)
# Same fixedQA reward config as comp03 (topk_qa=60, inner=32, terminal=128).
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_cont_lr1e6_topk50.sh > logs/<job_id>/32sess_cont_lr1e6_topk50_launch.log 2>&1 &

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
export INNER_N=4   # same as comp03

# KEY CHANGES vs comp03:
export STAGE_LR=1e-6       # was 2e-6 — gentler fine-tuning from strong checkpoint
export TOP_K_MEMORIES=50   # was 25 (default) — more memory candidates per query

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_cont_lr1e6_topk50"

export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
