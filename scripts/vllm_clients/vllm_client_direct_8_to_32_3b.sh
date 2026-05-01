#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# P7 fill: 3B 32-sess training via direct 8-sess→32-sess curriculum (skipping 16-sess).
# Closes the missing P7 cell — 3B was previously trained at 8-sess only (jetaoz29 val=0.424).
# Pairs with 7B direct_8_to_32 (test=0.501) and 7B 32sess_champion_v2 (test=0.498).
#
# Starting from: 3B 8sess champion (jetaoz29-equivalent, comp=0.2 step10) on H200.
# Same hyperparams as 7B direct_8_to_32 except model.
#
# Usage: srun --jobid=<H100_JID> --overlap -N1 -n1 bash vllm_client_direct_8_to_32_3b.sh

export SKIP_NODE_CHECK=1

# 3B 8sess champion (jetaoz29 equivalent: comp=0.2, 6turns, full step10)
export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_8sess_fix_answeragent__20260417_184031_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5
export COMPRESSION_PENALTY=0.3

export RUN_TAG="p7_3b_direct_8_to_32"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
