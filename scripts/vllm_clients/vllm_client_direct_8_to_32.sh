#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# G8 curriculum ablation: direct jump from 8-sess champion to 32-sess (skipping 16-sess)
# Tests whether the 16-sess curriculum step is necessary.
# Champion path: 8-sess → 16-sess → 32-sess. This skips the 16-sess stage.
#
# Comparison points:
#   - This run:         direct 8-sess champion → 32-sess → expected lower than champion_v2
#   - 32sess_champion_v2: full curriculum (8→16→32) → test/acc=0.501
#   - direct32sess (from base): crashed, test/acc=0.258
#
# Starting from: 8sess_champion step10 (turns6, comp0.2, thresh05)
# Note: 16-sess champion starts from 16sess_champion_v2 step5.
#       This script shows what happens if we skip 16-sess entirely.
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_direct_8_to_32.sh

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_8sess_8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5

export QA_TOP_K_PER_STAGE_OVERRIDE=30   # same as champion_v2 for fair comparison
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

export RUN_TAG="direct_8_to_32"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
