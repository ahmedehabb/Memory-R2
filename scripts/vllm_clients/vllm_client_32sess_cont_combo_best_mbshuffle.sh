#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess continuation from comp03 step5 — COMBO + PPO mini-batch shuffle experiment
#
# Goal: test whether randomizing mini-batch order inside PPO updates improves stability
#       by removing session-order bias from the final mega-batch.
#
# Start: 32sess_fixedqa_comp03 step5 (val=0.491, mfail=0.067, test=0.498)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_cont_combo_best_mbshuffle.sh > logs/<job_id>/32sess_cont_combo_best_mbshuffle_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_fixedqa_comp03__20260402_233703_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# New training-mechanics ablation key
export MINI_BATCH_SHUFFLE=True

# Fixed QA coverage (same as comp03)
export QA_TOP_K_PER_STAGE_OVERRIDE=60
export REMA_REWARD_MAX_QA_TRAIN_INNER=32
export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=128

export COMPRESSION_PENALTY=0.3
export INNER_GPRO_FRAC=0.5

# Match cont_combo_best hyperparams for controlled comparison
export INNER_N=8
export STAGE_LR=1e-6
export TOP_K_MEMORIES=50

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_cont_combo_best_mbshuffle"

# H100-safe memory profile for relaunches if needed.
export VLLM_GPU_MEMORY_UTILIZATION=0.5
export VLLM_MAX_NUM_BATCHED_TOKENS=32768

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
