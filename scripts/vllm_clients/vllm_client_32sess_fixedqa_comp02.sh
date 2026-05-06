#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess FIXED HYPERPARAMS run 1 — comp=0.2, topk=60, inner_qa=32, terminal_qa=128
# KEY FIXES vs all prior 32-sess runs:
#   - QA_TOP_K_PER_STAGE_OVERRIDE=60  (was 30: 2.4% coverage → now 4.8%)
#   - REMA_REWARD_MAX_QA_TRAIN_INNER=32  (was 16: 1.3% → 2.6% QA coverage inner GRPO)
#   - REMA_REWARD_MAX_QA_TRAIN_TERMINAL=128  (was 64: 5.1% → 10.2% coverage terminal)
#   - COMPRESSION_PENALTY=0.2  (was 0.3: lower to prevent collapse at higher topk)
#   - max_response_length=4096  (was 2048: full 32k context)
# Start: 16sess_champion_v2 step5 (val=0.488, mfail=0.067)
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_fixedqa_comp02.sh > logs/<job_id>/32sess_fixedqa_comp02_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="<repo>/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn

# Fixed QA coverage
export QA_TOP_K_PER_STAGE_OVERRIDE=60
export REMA_REWARD_MAX_QA_TRAIN_INNER=32
export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=128

export COMPRESSION_PENALTY=0.2
export INNER_GPRO_FRAC=0.5

export STAGES_OVERRIDE=32
export EPOCHS_PER_STAGE_OVERRIDE=5

export RUN_TAG="32sess_fixedqa_comp02"

# OOM fix: lower vLLM GPU memory utilization so PyTorch has more headroom for loss.backward()
# 0.8 → vLLM takes ~111GB, leaving only ~28GB → OOM. 0.6 → ~84GB vLLM, ~55GB PyTorch = safe.
export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
