#!/usr/bin/env bash
set -euo pipefail

# Generic VERL PPO trainer (non-ReMA) with custom F1 reward on <answer> tags.
# 1) Build parquet first with:
#    python sft/prepare_rlhf_from_traces.py --inputs data/sft/answer_traces_*.jsonl
# 2) Run this script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Make launcher robust even if editable installs are not active in this shell.
export PYTHONPATH="$REPO_ROOT/src/verl${PYTHONPATH:+:$PYTHONPATH}"

TRAIN_FILE="${TRAIN_FILE:-data/sft_rlhf/f1_gt_025/train.parquet}"
VAL_FILE="${VAL_FILE:-data/sft_rlhf/f1_gt_025/val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"
TRAIN_BS="${TRAIN_BS:-64}"
PPO_MINI_BS="${PPO_MINI_BS:-16}"
PPO_MICRO_BS_PER_GPU="${PPO_MICRO_BS_PER_GPU:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-12288}"
MAX_RESP_LEN="${MAX_RESP_LEN:-1024}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-auto}"
FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
EXP_NAME="${EXP_NAME:-normal_answer_f1_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rema-normal-trainer}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rema}"
CONDA_ROOT="${CONDA_ROOT:-<workspace>/miniconda3}"

REWARD_PATH="$REPO_ROOT/sft/rewards/answer_tag_f1_reward.py"

# Workaround: importing reward_manager package triggers ReMA judge_llm config checks.
# Normal trainer path here does not use judge_llm, but we set a safe default provider
# to satisfy import-time validation.
export JUDGE_PROVIDER="${JUDGE_PROVIDER:-together}"
export TOGETHER_API_KEY="${TOGETHER_API_KEY:-DUMMY_KEY_NOT_USED}"
export TOGETHER_NO_BANNER="${TOGETHER_NO_BANNER:-1}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[normal_trainer_answer_f1] missing TRAIN_FILE: $TRAIN_FILE"
  exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "[normal_trainer_answer_f1] missing VAL_FILE: $VAL_FILE"
  exit 1
fi
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "[normal_trainer_answer_f1] missing reward file: $REWARD_PATH"
  exit 1
fi

# Keep PPO mini-batch valid for quick smoke runs where TRAIN_BS may be small.
if (( PPO_MINI_BS > TRAIN_BS )); then
  echo "[normal_trainer_answer_f1] PPO_MINI_BS ($PPO_MINI_BS) > TRAIN_BS ($TRAIN_BS), clamping to TRAIN_BS"
  PPO_MINI_BS="$TRAIN_BS"
fi

# For single-GPU runs with a 7B model, conservative PPO batches avoid optimizer OOM.
if (( N_GPUS == 1 )) && (( PPO_MINI_BS > 16 )); then
  echo "[normal_trainer_answer_f1] single-GPU run detected, clamping PPO_MINI_BS to 16 (was $PPO_MINI_BS)"
  PPO_MINI_BS=16
fi

if [[ "$FSDP_OPTIMIZER_OFFLOAD" == "auto" ]]; then
  if (( N_GPUS == 1 )); then
    FSDP_OPTIMIZER_OFFLOAD=True
  else
    FSDP_OPTIMIZER_OFFLOAD=False
  fi
fi

echo "[normal_trainer_answer_f1] train=$TRAIN_FILE"
echo "[normal_trainer_answer_f1] val=$VAL_FILE"
echo "[normal_trainer_answer_f1] model=$MODEL_PATH"
echo "[normal_trainer_answer_f1] exp=$EXP_NAME"
echo "[normal_trainer_answer_f1] ppo_mini_bs=$PPO_MINI_BS ppo_micro_bs_per_gpu=$PPO_MICRO_BS_PER_GPU"
echo "[normal_trainer_answer_f1] fsdp_optimizer_offload=$FSDP_OPTIMIZER_OFFLOAD fsdp_param_offload=$FSDP_PARAM_OFFLOAD ref_param_offload=$REF_PARAM_OFFLOAD"

PYTHON_RUNNER=(python)
if [[ -n "$CONDA_ENV_NAME" ]]; then
  # Prefer activating the env so all child processes inherit the same runtime.
  if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
    if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$CONDA_ROOT/etc/profile.d/conda.sh"
    fi
    if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
      echo "[normal_trainer_answer_f1] warning: conda activate $CONDA_ENV_NAME failed, using conda run fallback"
      PYTHON_RUNNER=(conda run -n "$CONDA_ENV_NAME" python)
    fi
  fi

  # Keep PATH aligned even when activation was skipped (already active shell).
  if [[ -d "$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin" ]]; then
    export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:$PATH"
  fi
fi

PYTHONUNBUFFERED=1 "${PYTHON_RUNNER[@]}" -m verl.trainer.main_ppo \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  data.train_batch_size="$TRAIN_BS" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LEN" \
  actor_rollout_ref.rollout.response_length="$MAX_RESP_LEN" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BS" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BS_PER_GPU" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.optimizer_offload="$FSDP_OPTIMIZER_OFFLOAD" \
  critic.model.fsdp_config.param_offload="$FSDP_PARAM_OFFLOAD" \
  reward_model.enable=False \
  reward_model.reward_manager=naive \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","wandb"]'
