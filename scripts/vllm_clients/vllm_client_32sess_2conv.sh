#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# E3: 32-sess training with 2 train conversations (reduced gradient variance).
#
# Motivation: At 8-sess, 2conv training gave val=0.496 vs 1conv val=0.403 (+0.093).
# At 32-sess we always used 1 conv. With 2 train convs, gradient variance is halved
# — the reward signal from 2 independent conversations is more reliable even with
# sparse topk=30 coverage. This could enable the model to learn cross-session patterns
# that 1-conv training misses.
#
# LoCoMo has exactly 2 training conversations: conv-43 (29 sessions) and conv-47.
# Both will be used for training. Val and test remain the same fixed splits.
#
# Comparison target: 32sess_champion_v2 1conv (test=0.501), 8-sess champion (test=0.496)
# If 2conv 32-sess reaches test≥0.508, curriculum benefit vs 8-sess becomes clear (+0.012+).
#
# Usage: srun --jobid=<H200> --overlap -N1 -n1 bash vllm_client_32sess_2conv.sh

export SKIP_NODE_CHECK=1

# Start from 16-sess champion_v2 (same as champion_v2 path)
export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess settings
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

# KEY CHANGE: use 2 train conversations with 8 rollouts (matches 8-sess 2conv config)
# standalone.sh reads NUM_TRAIN_CONVS and num_rollouts directly
export NUM_TRAIN_CONVS=2
export num_rollouts=8   # 2conv × 8rollouts = 16 total (same budget as 1conv×16)

export RUN_TAG="32sess_2conv"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
