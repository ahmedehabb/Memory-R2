#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# N7 comp=0.1 32-sess. Fills the missing low-but-not-zero point in the comp sweep.
# Companion to: vllm_client_32sess_comp02.sh (=0.2), comp03.sh (=0.3), comp04.sh (=0.4)
# Continues 16->32 from 16sess_champion_v2 ckpt (same warmup as N7 comp=0/0.5).
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_comp01.sh > logs/<job_id>/32sess_comp01_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.1

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="n7_comp01_32sess"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
