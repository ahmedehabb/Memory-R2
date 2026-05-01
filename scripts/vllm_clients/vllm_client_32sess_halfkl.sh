#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# P4: 32-sess training with HALF KL penalty (kl_loss_coef=0.0005 vs default 0.001).
#
# Motivation: Reviewer concern — KL penalty at curriculum stage transitions may be
# too aggressive. When transitioning from 16-sess to 32-sess, the policy has already
# drifted from the Qwen base. A large KL coef (0.001) constrains the policy hard
# right when it needs to adapt to 4x more sessions. Halving KL allows freer
# exploration of 32-session memory strategies.
#
# Hypothesis: Lower KL → better policy adaptation → higher accuracy than champion_v2.
# Risk: Without KL regularization, policy may collapse (like fixedqa_comp02 at step3).
#
# Start: 16-sess champion_v2 step5 (val=0.488, mfail=0.067) — same as champion_v2.
# Config: Same as champion_v2 except kl_loss_coef=0.0005 (half).
# Compare: champion_v2 (kl=0.001, val=0.523, test=0.501).
#
# Usage: srun --jobid=<H200> --overlap -N1 -n1 bash vllm_client_32sess_halfkl.sh

export SKIP_NODE_CHECK=1

# Same start point as champion_v2
export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess settings identical to champion_v2
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

# KEY CHANGE: Half KL coef to test gentler regularization at stage transition
export kl_loss_coef=0.0005   # default=0.001; halved to reduce constraint on 16→32 transition

export RUN_TAG="32sess_halfkl"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
