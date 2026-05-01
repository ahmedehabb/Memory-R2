#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 16-sess training with QA_TOP_K=85 (proportionally correct: ~17% coverage of ~520 items)
# Ablation vs 16sess_champion_v2 (same config, but top_k=30 → 5.7% coverage)
# Starting from: turns6_comp02_thresh05 8-sess step10 checkpoint
#
# Usage: srun --jobid=<job_id> bash vllm_client_16sess_topk85.sh

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_8sess_8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed"

# 16-sess config matching champion_v2 (6 turns, comp=0.2, clip=0.1, kl=0.001)
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.1
export CLIP_MODE=turn
export AGG_MODE=turn

# KEY CHANGE: top_k=108 gives ~16.4% coverage of ~657 16-sess peak memory items
# Proportionally matches 8-sess: top_k=30 / ~183 items = 16.3%
# vs top_k=30 = 4.6% used in champion_v2
export QA_TOP_K_PER_STAGE_OVERRIDE=108

export STAGES_OVERRIDE=16
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="16sess_topk108"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
