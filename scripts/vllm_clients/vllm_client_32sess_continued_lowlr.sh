#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# E2: 32-sess continued training from fixedqa_comp03 step5, with much lower LR.
#
# Motivation: The original 32-sess training used LR=2e-6. When extended, the policy
# collapsed after 5 extra steps (val dropped from 0.484 → 0.175, mfail→0.161).
# With LR=5e-7 (4x lower), policy updates are smaller, gradient noise from sparse
# reward (~2.4% topk coverage) does less damage, and training can continue safely.
#
# Start model: 32sess_fixedqa_comp03 step5 (val=0.491, mfail=0.067 — very stable)
# This is the safest 32-sess checkpoint: lowest mfail (0.067 < champion_v2's 0.105).
#
# Paper argument: if this reaches test/acc≥0.508, the gap vs 8-sess becomes +0.012+
# which is more convincing for curriculum. Even if it stays flat, stability under
# continued training demonstrates the curriculum stabilized the policy.
#
# Usage: srun --jobid=<H200> --overlap -N1 -n1 bash vllm_client_32sess_continued_lowlr.sh

export SKIP_NODE_CHECK=1

# Resume from fixedqa_comp03 step5
export CURRENT_MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_fixedqa_comp03__20260402_233703_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess settings (same as champion)
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

# KEY CHANGE: Much lower LR to prevent collapse on continued training
export STAGE_LR=5e-7   # 4x lower than default 2e-6

export RUN_TAG="32sess_continued_lowlr"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
