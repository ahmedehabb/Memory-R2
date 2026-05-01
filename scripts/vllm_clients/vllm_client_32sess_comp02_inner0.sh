#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess: comp=0.2 + inner_grpo_frac=0.0 combined
# Hypothesis: lower comp penalty (0.2) prevents memory collapse,
#             AND removing inner GRPO reduces mfail (inner0 had 0.038@step4 vs topk80 0.074)
# Baseline: champion_v2 (comp=0.3, inner=0.5, topk=30) val=0.466, mfail=0.105
# Comp02:   (comp=0.2, inner=0.5, topk=30) — running on 3942475
# Inner0:   (comp=0.3, inner=0.0, topk=80) — step 4/5, mfail=0.038
# This:     (comp=0.2, inner=0.0, topk=30) — combining best of both
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_comp02_inner0.sh > logs/<job_id>/32sess_comp02_inner0_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.2
export INNER_GPRO_FRAC=0.0

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_comp02_inner0"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
