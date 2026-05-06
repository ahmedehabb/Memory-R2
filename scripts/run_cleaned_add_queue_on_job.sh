#!/bin/bash
set -euo pipefail

# Automated cleaned LongMemEval add-stage queue (resume-safe).
# Intended usage:
#   srun --jobid=<H100_JOB_ID> --overlap -N1 -n1 bash scripts/run_cleaned_add_queue_on_job.sh
#
# Behavior:
# - Runs missing add-stage combinations sequentially.
# - Starts a dedicated memory-model vLLM server per tier (base/8sess/16sess/32sess).
# - Uses tier-correct memory models (never gpt-oss fallback by design).
# - Skips combos that are already complete by pkl count.

REPO_DIR="<repo>"
PY_BIN="<workspace>/miniconda3/envs/rema/bin/python"
OPENAI_CACHE_DIR="<workspace>/openai_cache"
RESULTS_DIR="${REPO_DIR}/testing/results"
PIPE_DIR="${REPO_DIR}/testing/pipeline_test_longmemeval"
RENDEZVOUS_DIR="${REPO_DIR}/vllm_servers_cleaned_queue"
SERVER_SCRIPT="${REPO_DIR}/vllm_server_standalone.sh"
LOG_DIR="${REPO_DIR}/logs/cleaned_add_queue"

mkdir -p "${RESULTS_DIR}" "${RENDEZVOUS_DIR}" "${LOG_DIR}"

declare -A MODEL_PATH
MODEL_PATH[base]="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH[8sess]="${REPO_DIR}/checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_10/hf_fixed"
MODEL_PATH[16sess]="${REPO_DIR}/checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
MODEL_PATH[32sess]="${REPO_DIR}/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"

declare -A MAX_TURNS
MAX_TURNS[base]=4
MAX_TURNS[8sess]=4
MAX_TURNS[16sess]=6
MAX_TURNS[32sess]=6

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "${RENDEZVOUS_DIR}/server_0.txt"
  SERVER_PID=""
}
trap cleanup_server EXIT

dataset_expected_count() {
  local dataset_json="$1"
  "${PY_BIN}" - <<PY
import json
with open("${dataset_json}", "r") as f:
    d = json.load(f)
print(len(d))
PY
}

current_pkl_count() {
  local mem_dir="$1"
  if [[ -d "${mem_dir}" ]]; then
    find "${mem_dir}" -maxdepth 1 -name '*.pkl' | wc -l
  else
    echo 0
  fi
}

start_server_for_tier() {
  local tier="$1"
  local model_path="${MODEL_PATH[$tier]}"
  local server_log="${LOG_DIR}/server_${tier}_$(date +%Y%m%d_%H%M%S).log"

  if [[ "${tier}" != "base" ]] && [[ ! -d "${model_path}" ]]; then
    echo "[queue] ERROR: model path missing for ${tier}: ${model_path}"
    return 1
  fi

  cleanup_server

  echo "[queue] Starting memory server for tier=${tier}"
  (
    export RENDEZVOUS_DIR="${RENDEZVOUS_DIR}"
    export SERVER_IDX=0
    export VLLM_PORT=${QUEUE_VLLM_PORT:-8119}
    # Use full 4-GPU node for add-stage serving (much faster than single-GPU server).
    export VLLM_TENSOR_PARALLEL=4
    export VLLM_MAX_MODEL_LEN=32768
    export VLLM_JUDGE_MODEL="${model_path}"
    bash "${SERVER_SCRIPT}"
  ) > "${server_log}" 2>&1 &
  SERVER_PID=$!

  # Wait up to 20 min for model load (large ckpts + cold cache need headroom)
  for _ in $(seq 1 600); do
    if [[ -f "${RENDEZVOUS_DIR}/server_0.txt" ]]; then
      local hp
      hp="$(cat "${RENDEZVOUS_DIR}/server_0.txt")"
      if curl -sf "http://${hp}/v1/models" >/dev/null; then
        echo "[queue] Server ready for ${tier} at http://${hp}/v1"
        return 0
      fi
    fi
    sleep 2
  done

  echo "[queue] ERROR: server did not become ready for tier=${tier}. Log: ${server_log}"
  return 1
}

run_add_combo() {
  local dataset_tag="$1"  # longmemeval_s_cleaned / longmemeval_m_cleaned
  local tier="$2"         # base / 8sess / 16sess / 32sess

  local dataset_json="${PIPE_DIR}/dataset/${dataset_tag}.json"
  local mem_dir="${RESULTS_DIR}/${dataset_tag}_${tier}_memory"
  local add_log="${RESULTS_DIR}/${dataset_tag}_${tier}_add_auto.log"
  local expected
  expected="$(dataset_expected_count "${dataset_json}")"
  local have
  have="$(current_pkl_count "${mem_dir}")"

  echo "[queue] >>> ${dataset_tag} x ${tier}: have=${have}, expected=${expected}"
  if [[ "${have}" -ge "${expected}" ]]; then
    echo "[queue] SKIP ${dataset_tag} x ${tier} (already complete)"
    return 0
  fi

  start_server_for_tier "${tier}"
  local hp
  hp="$(cat "${RENDEZVOUS_DIR}/server_0.txt")"
  local mem_url="http://${hp}/v1"
  local served_model
  served_model="$("${PY_BIN}" - <<PY
import requests
url="${mem_url}/models"
data=requests.get(url, timeout=15).json().get("data", [])
print(data[0]["id"] if data else "")
PY
)"
  if [[ -z "${served_model}" ]]; then
    echo "[queue] ERROR: could not resolve served model id from ${mem_url}/models"
    return 1
  fi
  if [[ "${tier}" == "base" ]] && [[ "${served_model}" == *"gpt-oss"* ]]; then
    echo "[queue] ERROR: base tier resolved to gpt-oss (${served_model}) which is invalid for memory add-stage."
    return 1
  fi

  echo "[queue] RUN  ${dataset_tag} x ${tier} using ${mem_url} model_id=${served_model}"
  (
    cd "${PIPE_DIR}"
    export OPENAI_CACHE_DIR="${OPENAI_CACHE_DIR}"
    export REMA_LONGMEMEVAL_FORCE_REPROCESS=0
    export REMA_LONGMEMEVAL_MAX_NUM_TURNS="${MAX_TURNS[$tier]}"
    "${PY_BIN}" run_experiments.py \
      --method rema_add \
      --dataset_path "dataset/${dataset_tag}.json" \
      --memExtractor_url "${mem_url}" \
      --memExtractor_model "${served_model}" \
      --memAgent_url "${mem_url}" \
      --memAgent_model "${served_model}" \
      --memory_store_dir "${mem_dir}" \
      --embedding_cache_dir "${OPENAI_CACHE_DIR}"
  ) 2>&1 | tee "${add_log}"

  have="$(current_pkl_count "${mem_dir}")"
  echo "[queue] DONE ${dataset_tag} x ${tier}: now have=${have}/${expected}"
  cleanup_server
}

echo "[queue] Host: $(hostname)"
echo "[queue] Starting cleaned add-stage queue at $(date)"
STATUS_FILE="${RESULTS_DIR}/cleaned_add_queue_status_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "timestamp\tdataset\ttier\tstatus\tnote" > "${STATUS_FILE}"

# Priority order: s_cleaned first (all tiers), then m_cleaned.
for dataset_tag in longmemeval_s_cleaned longmemeval_m_cleaned; do
  for tier in base 8sess 16sess 32sess; do
    ts="$(date +%F' '%T)"
    if run_add_combo "${dataset_tag}" "${tier}"; then
      echo -e "${ts}\t${dataset_tag}\t${tier}\tOK\tcompleted_or_skipped" >> "${STATUS_FILE}"
    else
      echo -e "${ts}\t${dataset_tag}\t${tier}\tFAIL\tcheck_logs" >> "${STATUS_FILE}"
      echo "[queue] WARN: ${dataset_tag} x ${tier} failed; continuing to next combo."
      cleanup_server
      continue
    fi
  done
done

echo "[queue] Queue completed at $(date). Status file: ${STATUS_FILE}"
