#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess continuation from champion_v2 step5 — fixedQA + inner_n=8 + topk_memories=50
#
# Goal: apply the best reward signal (fixedQA: topk_qa=60, inner=32, terminal=128) and
#       best retrieval (topk_mem=50) to the STRONGEST known checkpoint: champion_v2
#       (test/acc=0.501, the current best). champion_v2 had mfail=0.105 (worse than
#       comp03's 0.067), meaning its memory management still has room to improve.
#       Combining: best base model + best reward quality + inner_n=8 + topk_mem=50.
#
# Why NOT starting from comp03: comp03 test=0.498 < champion_v2 test=0.501. Starting
# from the stronger base gives the best chance of exceeding 0.501.
#
# Start: 32sess_champion_v2 step5 (val=0.466, mfail=0.105, test=0.501 — CURRENT BEST)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_cont_from_champ_fixedqa.sh > logs/<job_id>/32sess_cont_from_champ_fixedqa_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# Fixed QA coverage — key upgrade vs champion_v2's original FAST config (topk=30, inner=16, terminal=64)
export QA_TOP_K_PER_STAGE_OVERRIDE=60
export REMA_REWARD_MAX_QA_TRAIN_INNER=32
export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=128

export COMPRESSION_PENALTY=0.3   # keep stable — DO NOT use 0.2 (memory collapse risk)
export INNER_GPRO_FRAC=0.5

# KEY CHANGES vs champion_v2's original 32-sess training:
export INNER_N=8            # was 4 (proven: inner_n8 gave +0.021 at 16-sess)
export STAGE_LR=1e-6        # conservative LR — champion_v2 had mfail=0.105, be careful
export TOP_K_MEMORIES=50    # was 25 (default) — more retrieval context

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_cont_from_champ_fixedqa"

export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
