#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess EXTENDED training from 32sess_champion_v2 final ckpt (step5, val=0.466)
# Same config as champion_v2 — test if more training steps improve 32-sess val
# 8-sess precedent: turns6 improved from step5(0.467) to step10(0.498) = +0.031
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_extended.sh > logs/<job_id>/32sess_extended_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

# Start from 32sess_champion_v2 step5 (val=0.466, mfail=0.105)
export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.3

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_extended"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
