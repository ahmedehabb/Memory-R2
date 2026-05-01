#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 32-sess extended training from champion_v2 WITH fixed hyperparams
# Start: 32sess_champion_v2 step5 (val=0.466, mfail=0.105)
# Fixes applied: topk=60, inner_qa=32, terminal_qa=128, comp=0.2, response=4096
# Previous extended run (vllm_client_32sess_extended.sh) used old hyperparams → killed.
#
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_32sess_extended_fixedqa.sh > logs/<job_id>/32sess_extended_fixedqa_launch.log 2>&1 &

export SKIP_NODE_CHECK=1

export CURRENT_MODEL_PATH_OVERRIDE="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

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

export RUN_TAG="32sess_extended_fixedqa"

# Fix for KV cache OOM on nodes with leftover CUDA memory from previous crashes
# Lower gpu_memory_utilization so PyTorch actor has enough for loss.backward() (was 0.8 → OOM)
export VLLM_GPU_MEMORY_UTILIZATION=0.6

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
