#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess inner0 ablation (Priority B3)
# Tests whether inner GRPO helps at 32-sess (inner_grpo_frac=0.0 vs 0.5)
# Same config as 32sess_topk80 except inner_grpo_frac=0.0
# Starting from: 16sess_champion_v2 step5 checkpoint
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_inner0.sh

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.0   # KEY: no inner GRPO sampling (ablation vs topk80 which uses 0.5)

export QA_TOP_K_PER_STAGE_OVERRIDE=80   # same as topk80 for fair comparison
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

export RUN_TAG="32sess_inner0"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
