#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess training with COMPRESSION_PENALTY=0.2 (vs 0.3 in champion_v2/topk80/topk120)
# Goal: test if lower comp penalty prevents the aggressive memory collapse seen in topk80
# topk80 showed mem 1267→481 over 5 steps with mfail spike to 0.285 at step5
# comp=0.2 matches 16-sess champion_v2 setting — keeps memory larger → more retrieval surface
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_comp02.sh > logs/<job_id>/32sess_comp02_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess config matching champion_v2 (6 turns, clip=0.2, kl=0.001)
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# KEY CHANGE: comp=0.2 instead of 0.3 (same as 16-sess champion_v2)
# topk=30 to match champion_v2 baseline (known to work)
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.2

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_comp02"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
