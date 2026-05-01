#!/usr/bin/env bash
# N11 SFT-answer-agent CONTINUATION from step 50 of normal_answer_f1_thr015 run.
# Goal: keep training the SFT answer agent past step 50 to see if open-domain
# accuracy keeps improving (was still rising at step 50 per program.md N10/N11).
#
# Continuation policy: load the last HF-exported checkpoint as the new base model,
# then run normal_trainer_answer_f1.sh for 5 more epochs (TOTAL_EPOCHS=10 cumulative).
#
# Usage (on a free 4-GPU H200 allocation):
#   srun --jobid=<job_id> --overlap -N1 -n1 bash scripts/rl/n11_sft_answer_continuation.sh > logs/<job_id>/n11_sft_answer_cont_launch.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Continue from the prior step-50 HF export.
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/outputs/answer_agent_sft_hf/normal_answer_f1_thr015_testfreq5_step50_20260410_150053}"

# Same data as the original run (F1>=0.15 threshold split is the strongest one).
export TRAIN_FILE="${TRAIN_FILE:-data/sft_rlhf/f1_gt_015/train.parquet}"
export VAL_FILE="${VAL_FILE:-data/sft_rlhf/f1_gt_015/val.parquet}"

# 4 H200 GPUs, train 5 more epochs, save every 5 steps.
export N_GPUS="${N_GPUS:-4}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export TRAIN_BS="${TRAIN_BS:-64}"
export PPO_MINI_BS="${PPO_MINI_BS:-16}"
export PPO_MICRO_BS_PER_GPU="${PPO_MICRO_BS_PER_GPU:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

export EXP_NAME="${EXP_NAME:-n11_sft_answer_cont_from_step50_$(date +%Y%m%d_%H%M%S)}"
export PROJECT_NAME="${PROJECT_NAME:-rema-normal-trainer}"

bash "$SCRIPT_DIR/normal_trainer_answer_f1.sh"
