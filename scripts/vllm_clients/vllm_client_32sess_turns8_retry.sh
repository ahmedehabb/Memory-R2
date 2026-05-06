#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# N8 turns=8 32-sess RETRY (the previous attempt on 3985703 failed with
# "Could not find checkpoint for stage 32"). Per program.md:40 documented
# fallback: "If not present, continue 16->32 from the 8-sess turns=N ckpts
# (already trained)." We deliberately SKIP the 16-sess turns=8 ckpt because
# that run COLLAPSED (val=0.163, see program.md:77), so warming from it would
# just propagate collapse.
#
# Warmup: 8-sess turns=8 step10 hf_fixed
# Conversion is run separately (see logs/convert_8sess_turns8_step10_hf_fixed.log)
# and must complete before the trainer reads the path on first step.
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_turns8_retry.sh > logs/<job_id>/n8_turns8_retry_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_8sess_3975036_8turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed"

# 32-sess config matching champion_v2 stack except turns=8
export MAX_NUM_TURNS=8
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

export QA_TOP_K_PER_STAGE_OVERRIDE=30
export COMPRESSION_PENALTY=0.3

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="n8_turns8_32sess_retry_from8s"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
