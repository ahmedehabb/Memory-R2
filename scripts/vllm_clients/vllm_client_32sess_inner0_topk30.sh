#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# G6 ablation: 32-sess inner0 with topk=30 (matched to champion_v2)
# Tests inner GRPO effect at 32-sess with SAME topk=30 as champion_v2.
# This gives the clean, unconfounded inner GRPO gap at 32-sess.
#
# Comparison points:
#   - This run:         inner=0.0, topk=30, comp=0.3 → should show ~test/acc between 0.365 and 0.501
#   - 32sess_champion_v2: inner=0.5, topk=30, comp=0.3 → test/acc=0.501
#   - 32sess_inner0 (old): inner=0.0, topk=80, comp=0.3 → test/acc=0.365
#   - 32sess_topk80:    inner=0.5, topk=80, comp=0.3 → test/acc=0.460
#
# Starting from: 16sess_champion_v2 step5 checkpoint (same as 32sess_champion_v2)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_inner0_topk30.sh

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.0   # KEY: no inner GRPO (ablation — matched topk=30 with champion_v2)

export QA_TOP_K_PER_STAGE_OVERRIDE=30   # KEY: topk=30, matched to champion_v2 (not 80 like old inner0)
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

export RUN_TAG="32sess_inner0_topk30"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
