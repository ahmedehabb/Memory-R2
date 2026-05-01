#!/usr/bin/env bash
# N11 SFT-answer-agent per-checkpoint eval.
# Convert one VERL checkpoint of the SFT-answer trainer to HF, then run a
# qwen-judge test-eval where the SFT-answer model itself is served as the
# answer agent + judge — same protocol as the existing
# qwen_judge_*_sft_answeragent_* runs (see logs/3963648/).
#
# Required env:
#   SFT_CKPT_STEP=<5|10|15|20|25|30|35|40|45>  (which step to eval)
# Optional env:
#   SFT_RUN_DIR=<run dir under checkpoints/rema-normal-trainer/>
#       default: normal_answer_f1_thr015_4gpu_minibs16_testfreq5_job3955610_20260410_005700
#   RUN_TAG=<custom tag>  default: n11_sft_answeragent_step<STEP>_<ts>
#   MEM_CKPT_PATH=<rema-curriculum-v1 ckpt>  default: 32sess_champion_v2 step5 hf_fixed
#
# Usage:
#   srun --jobid=<job_id> --overlap -N1 -n1 \
#       env SFT_CKPT_STEP=20 \
#       bash scripts/rl/n11_sft_answer_eval_ckpt.sh \
#       > logs/<job_id>/n11_sft_answer_eval_step20_launch.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SFT_CKPT_STEP="${SFT_CKPT_STEP:?SFT_CKPT_STEP=<5|10|15|20|25|30|35|40|45> required}"
SFT_RUN_DIR="${SFT_RUN_DIR:-normal_answer_f1_thr015_4gpu_minibs16_testfreq5_job3955610_20260410_005700}"
SRC_VERL_CKPT="$REPO_ROOT/checkpoints/rema-normal-trainer/$SFT_RUN_DIR/global_step_${SFT_CKPT_STEP}"

if [[ ! -d "$SRC_VERL_CKPT" ]]; then
  echo "[n11_sft_eval] missing VERL ckpt: $SRC_VERL_CKPT"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
HF_OUT="$REPO_ROOT/outputs/answer_agent_sft_hf/n11_sft_step${SFT_CKPT_STEP}_${TS}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rema}"
CONDA_ROOT="${CONDA_ROOT:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3}"
if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" || true
fi
if [[ -d "$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin" ]]; then
  export PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin:$PATH"
fi

echo "[n11_sft_eval] Converting VERL ckpt -> HF: $SRC_VERL_CKPT -> $HF_OUT"
python "$REPO_ROOT/convert_fsdp_to_hf.py" \
    --fsdp_checkpoint_path "$SRC_VERL_CKPT/actor" \
    --huggingface_model_path "Qwen/Qwen2.5-7B-Instruct" \
    --output_path "$HF_OUT" \
    --world_size 4

# Use this newly converted SFT-answer model as the JUDGE/answer-agent
# (the qwen test eval client expects a Qwen-protocol vllm server already
# running and named in vllm_servers_qwen/server_*.txt). For autonomous
# dispatch, the operator daemon `auto_qwen_cycle.sh` already manages
# Qwen-judge servers — point it at $HF_OUT once available.

# Memory pipeline = 32sess_champion_v2 (the canonical mem-policy under test)
export MODEL_PATH_OVERRIDE="${MEM_CKPT_PATH:-$REPO_ROOT/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed}"

export RUN_TAG="${RUN_TAG:-n11_sft_answeragent_step${SFT_CKPT_STEP}_${TS}}"
export ANSWER_AGENT_MODEL_OVERRIDE="$HF_OUT"

echo "[n11_sft_eval] HF model ready: $HF_OUT"
echo "[n11_sft_eval] RUN_TAG=$RUN_TAG"
echo "[n11_sft_eval] Hand-off: arrange a Qwen-judge vllm server with model=$HF_OUT,"
echo "[n11_sft_eval]           then run scripts/vllm_clients/vllm_client_test_eval_qwen.sh"
