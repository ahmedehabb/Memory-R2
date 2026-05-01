#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# N8 turns=4 32-sess (paper N8 multi-turn baseline at 32-sess).
# Champion is turns=6 (val=0.466). turns=8/10 already covered (val=0.430/0.356).
# This fills the missing turns=4 row in the 32-sess multi-turn ablation table.
#
# Warmup: 16sess turns=4 inner_n8 step5 hf_fixed (val=0.493 at 16-sess, turns=4, comp=0.2, inner=0.5 n=8)
# Continues 16->32 with same other params as champion (comp=0.3 to match champion's 32-sess setup).

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_16sess_16sess_inner_n8_j3939305__20260401_052207_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

# 32-sess config matching champion_v2 stack except turns=4
export MAX_NUM_TURNS=4
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.3

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="n8_turns4_32sess"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
