#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess training with QA_TOP_K=200 (proportionally correct: ~15% coverage of ~1300 items)
# Ablation vs 32sess_topk80 (6.2% coverage) and 32sess_champion_v2 (2.3% coverage)
# Starting from: 16sess_champion_v2 step5 checkpoint
#
# Usage: srun --jobid=<job_id> bash vllm_client_32sess_topk200.sh

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess config matching champion_v2 (6 turns, comp=0.3, clip=0.2, kl=0.001)
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# KEY CHANGE: top_k=200 gives ~15.4% coverage of ~1300 32-sess memory items
# vs top_k=80 = 6.2% (topk80 run), vs top_k=30 = 2.3% (champion_v2)
export QA_TOP_K_PER_STAGE_OVERRIDE=120

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

# 32-sess needs higher compression penalty for stability
export COMPRESSION_PENALTY=0.3

export RUN_TAG="32sess_topk120"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
