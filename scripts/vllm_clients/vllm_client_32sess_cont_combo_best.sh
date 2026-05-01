#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess continuation from comp03 step5 — COMBO: inner_n=8 + LR=1e-6 + topk_memories=50
#
# Goal: combine the two most promising individual changes (Exp A and Exp B).
#       inner_n=8 improves inner GRPO quality (proven at 16-sess: +0.021).
#       LR=1e-6 reduces risk of collapse on continued training.
#       topk_mem=50 gives broader memory retrieval context.
#       This is the "best of everything" conservative run.
#
# Start: 32sess_fixedqa_comp03 step5 (val=0.491, mfail=0.067, test=0.498)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_cont_combo_best.sh > logs/<job_id>/32sess_cont_combo_best_launch.log 2>&1 &

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
export INNER_N=8            # was 4 (16-sess inner8 gave +0.021)
export STAGE_LR=1e-6        # was 2e-6 (gentler fine-tuning)
export TOP_K_MEMORIES=50    # was 25 default

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_cont_combo_best"

export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
