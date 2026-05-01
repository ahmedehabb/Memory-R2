#!/usr/bin/env bash
set -euo pipefail

# One-shot automation tick for Qwen/SFT eval queue.
# Intended to be triggered periodically by cron (no sleep loop here).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOCK_FILE="$REPO_ROOT/logs/auto_qwen_cycle.lock"
mkdir -p "$REPO_ROOT/logs"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[auto-qwen] another tick is running; exiting"
  exit 0
fi

JOBS=(3960067 3963648)

is_job_step_running() {
  local job_id="$1"
  squeue -s -j "$job_id" 2>/dev/null | awk '{print $1}' | grep -Eq "^${job_id}\.[0-9]+$"
}

latest_tag_for_job() {
  local job_id="$1"
  local tag_file="logs/${job_id}/latest_qwen_sft_run_tag.txt"
  if [[ -f "$tag_file" ]]; then
    tr -d '\n' < "$tag_file"
  fi
}

latest_eval_log_for_tag() {
  local job_id="$1"
  local tag="$2"
  ls -1t "logs/${job_id}/${tag}_"*.log 2>/dev/null | grep -v '_launch.log' | head -n1 || true
}

extract_metric_from_log() {
  local log_file="$1"
  local metric_name="$2"
  grep -E "wandb:\s+${metric_name}\s" "$log_file" | tail -n1 | awk '{print $NF}'
}

extract_wandb_id() {
  local log_file="$1"
  grep -E "wandb.ai/.*/runs/" "$log_file" | tail -n1 | sed -E 's#.*runs/([a-zA-Z0-9]+).*#\1#'
}

start_model_from_tag() {
  local tag="$1"
  case "$tag" in
    qwen_judge_direct32sess_sft_answeragent_*) echo "ckpt:direct32sess_step5" ;;
    qwen_judge_sft_answeragent_mem_champion_fix404_*) echo "ckpt:32sess_champion_v2_step5" ;;
    qwen_judge_16sess_champion_v2_sft_answeragent_*) echo "ckpt:16sess_champion_v2_step5" ;;
    qwen_judge_direct16sess_sft_answeragent_*) echo "ckpt:direct16sess_step5" ;;
    qwen_judge_16sess_inner_n8_sft_answeragent_*) echo "ckpt:16sess_inner_n8_step5" ;;
    qwen_judge_16sess_inner0_sft_answeragent_*) echo "ckpt:16sess_inner0_step5" ;;
    qwen_judge_baseline_definitive_sft_answeragent_*) echo "base_qwen" ;;
    *) echo "ckpt:unknown" ;;
  esac
}

append_program_event() {
  local message="$1"
  local heading="### Auto Queue Events"
  if ! grep -q "$heading" program.md; then
    {
      echo ""
      echo "$heading"
      echo ""
    } >> program.md
  fi
  echo "- $(date '+%Y-%m-%d %H:%M:%S'): $message" >> program.md
}

maybe_record_finished_for_job() {
  local job_id="$1"
  local tag
  tag="$(latest_tag_for_job "$job_id")"
  [[ -z "$tag" ]] && return 0

  local log_file
  log_file="$(latest_eval_log_for_tag "$job_id" "$tag")"
  [[ -z "$log_file" ]] && return 0

  if grep -q "$tag" results.tsv; then
    return 0
  fi

  local acc bleu mhop
  acc="$(extract_metric_from_log "$log_file" 'test/acc/locomo' || true)"
  bleu="$(extract_metric_from_log "$log_file" 'test/bleu/locomo' || true)"
  mhop="$(extract_metric_from_log "$log_file" 'test/multi_hop_f1' || true)"

  # Not finished yet.
  if [[ -z "$acc" || -z "$bleu" || -z "$mhop" ]]; then
    return 0
  fi

  local wandb_id start_model commit
  wandb_id="$(extract_wandb_id "$log_file" || true)"
  start_model="$(start_model_from_tag "$tag")"
  commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

  printf "%s %s 0.000 %0.3f 0.000 keep DONE: TEST (Qwen judge, SFT answer-agent server): \`%s\` on job %s completed with test/acc=%s, bleu=%s, multi_hop_f1=%s (wandb %s).\n" \
    "$commit" "$start_model" "$acc" "$tag" "$job_id" "$acc" "$bleu" "$mhop" "${wandb_id:-unknown}" >> results.tsv

  append_program_event "Completed ${tag} on job ${job_id}: acc=${acc}, bleu=${bleu}, mhop_f1=${mhop}, wandb=${wandb_id:-unknown}."
  echo "[auto-qwen] recorded completion for $tag"
}

is_tag_running_anywhere() {
  local tag="$1"
  local current_tag
  for job_id in "${JOBS[@]}"; do
    current_tag="$(latest_tag_for_job "$job_id")"
    if [[ "$current_tag" == "${tag}_"* ]] && is_job_step_running "$job_id"; then
      return 0
    fi
  done
  return 1
}

launch_on_idle_job() {
  local job_id="$1"

  # Queue: remaining SFT-judge items only.
  local keys=(
    "qwen_judge_16sess_inner0_sft_answeragent"
    "qwen_judge_baseline_definitive_sft_answeragent"
  )
  local paths=(
    "checkpoints/rema-curriculum-v1/curr_16sess_16sess_inner0_j3939306__20260401_011639_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_5/hf_fixed"
    ""
  )

  local i key model_path
  for i in "${!keys[@]}"; do
    key="${keys[$i]}"
    model_path="${paths[$i]}"

    # Skip completed variants.
    if grep -q "$key" results.tsv; then
      continue
    fi

    # Skip if currently running.
    if is_tag_running_anywhere "$key"; then
      continue
    fi

    local run_tag
    run_tag="${key}_$(date +%Y%m%d_%H%M%S)"

    mkdir -p "logs/${job_id}"
    echo "$run_tag" > "logs/${job_id}/latest_qwen_sft_run_tag.txt"

    if [[ -n "$model_path" ]]; then
      MODEL_PATH_OVERRIDE="$model_path" RUN_TAG="$run_tag" \
        srun --jobid="$job_id" --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_test_eval_qwen.sh \
        > "logs/${job_id}/${run_tag}_launch.log" 2>&1 &
    else
      RUN_TAG="$run_tag" \
        srun --jobid="$job_id" --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_test_eval_qwen.sh \
        > "logs/${job_id}/${run_tag}_launch.log" 2>&1 &
    fi

    append_program_event "Launched ${run_tag} on idle job ${job_id}."
    echo "[auto-qwen] launched $run_tag on job $job_id"
    return 0
  done

  echo "[auto-qwen] no pending queue item to launch on job $job_id"
}

main() {
  echo "[auto-qwen] tick start $(date '+%F %T')"

  local job_id
  for job_id in "${JOBS[@]}"; do
    maybe_record_finished_for_job "$job_id"
  done

  for job_id in "${JOBS[@]}"; do
    if ! is_job_step_running "$job_id"; then
      launch_on_idle_job "$job_id"
    fi
  done

  echo "[auto-qwen] tick end $(date '+%F %T')"
}

main "$@"
