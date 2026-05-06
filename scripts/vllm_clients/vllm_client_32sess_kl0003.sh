#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess with kl_loss_coef=0.003 (3x standard 0.001)
# Higher KL penalty keeps model closer to reference → may reduce mfail spike at step5
# Baseline: champion_v2 (kl=0.001) val=0.466, mfail=0.105 (spike at step3+)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_kl0003.sh > logs/<job_id>/32sess_kl0003_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.3
export kl_loss_coef=0.003

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_kl0003"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
