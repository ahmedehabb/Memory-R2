#!/bin/bash
# Run ONE LME s_cleaned tier add-stage on a single SLURM node.
# Used to parallelize tiers across multiple GPU nodes.
# Usage:
#   srun --jobid=<JID> --overlap -N1 -n1 bash scripts/run_lme_tier.sh <TIER> <PORT> <RENDEZVOUS_SUBDIR>
# Examples:
#   bash scripts/run_lme_tier.sh 8sess 8121 lme_8sess
#   bash scripts/run_lme_tier.sh 16sess 8122 lme_16sess
#   bash scripts/run_lme_tier.sh 32sess 8123 lme_32sess

set -euo pipefail

TIER=$1
PORT=$2
RVDZ_SUBDIR=$3

REPO_DIR="<repo>"
PY_BIN="<workspace>/miniconda3/envs/rema/bin/python"
OPENAI_CACHE_DIR="<workspace>/openai_cache"
RESULTS_DIR="${REPO_DIR}/testing/results"
PIPE_DIR="${REPO_DIR}/testing/pipeline_test_longmemeval"
RENDEZVOUS_DIR="${REPO_DIR}/vllm_servers_${RVDZ_SUBDIR}"
SERVER_SCRIPT="${REPO_DIR}/vllm_server_standalone.sh"
LOG_DIR="${REPO_DIR}/logs/lme_per_tier"

mkdir -p "${RESULTS_DIR}" "${RENDEZVOUS_DIR}" "${LOG_DIR}"
rm -f "${RENDEZVOUS_DIR}/server_0.txt"

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

MODEL="${MODEL_PATH[$TIER]}"
TURNS="${MAX_TURNS[$TIER]}"
MEM_DIR="${RESULTS_DIR}/longmemeval_m_cleaned_${TIER}_memory"
DATASET="${PIPE_DIR}/dataset/longmemeval_m_cleaned.json"
ADD_LOG="${RESULTS_DIR}/longmemeval_m_cleaned_${TIER}_add_parallel_$(date +%Y%m%d_%H%M%S).log"
SERVER_LOG="${LOG_DIR}/server_${TIER}_$(date +%Y%m%d_%H%M%S).log"

# Skip if already complete
EXPECTED=$("${PY_BIN}" -c "import json; print(len(json.load(open('${DATASET}'))))")
if [[ -d "${MEM_DIR}" ]]; then
  HAVE=$(find "${MEM_DIR}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
else
  HAVE=0
fi
echo "[lme/${TIER}] have=${HAVE}/${EXPECTED}"
if [[ "${HAVE}" -ge "${EXPECTED}" ]]; then
  echo "[lme/${TIER}] SKIP — already complete"
  exit 0
fi

# Start mem-agent vLLM server (TP=4, blocking until ready)
echo "[lme/${TIER}] starting server on port ${PORT} for ${MODEL}"
(
  export RENDEZVOUS_DIR
  export SERVER_IDX=0
  export VLLM_PORT="${PORT}"
  export VLLM_TENSOR_PARALLEL=4
  export VLLM_MAX_MODEL_LEN=32768
  export VLLM_JUDGE_MODEL="${MODEL}"
  bash "${SERVER_SCRIPT}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Wait up to 20 min for server ready
for _ in $(seq 1 600); do
  if [[ -f "${RENDEZVOUS_DIR}/server_0.txt" ]]; then
    HP="$(cat "${RENDEZVOUS_DIR}/server_0.txt")"
    if curl -sf "http://${HP}/v1/models" >/dev/null; then
      echo "[lme/${TIER}] server ready at http://${HP}"
      break
    fi
  fi
  sleep 2
done

if [[ -z "${HP:-}" ]]; then
  echo "[lme/${TIER}] ERROR server not ready"
  kill -9 "${SERVER_PID}" 2>/dev/null || true
  exit 1
fi

# Resolve served model id
SERVED_MODEL="$("${PY_BIN}" - <<PY
import requests
print(requests.get("http://${HP}/v1/models", timeout=15).json()["data"][0]["id"])
PY
)"
echo "[lme/${TIER}] served_model=${SERVED_MODEL}"

# Run add-stage
(
  cd "${PIPE_DIR}"
  export OPENAI_CACHE_DIR
  export REMA_LONGMEMEVAL_FORCE_REPROCESS=0
  export REMA_LONGMEMEVAL_MAX_NUM_TURNS="${TURNS}"
  "${PY_BIN}" run_experiments.py \
    --method rema_add \
    --dataset_path "${DATASET}" \
    --memExtractor_url "http://${HP}/v1" \
    --memExtractor_model "${SERVED_MODEL}" \
    --memAgent_url "http://${HP}/v1" \
    --memAgent_model "${SERVED_MODEL}" \
    --memory_store_dir "${MEM_DIR}" \
    --embedding_cache_dir "${OPENAI_CACHE_DIR}" \
    --max_workers "${LME_MAX_WORKERS:-8}"
) 2>&1 | tee "${ADD_LOG}"

ADD_RC=${PIPESTATUS[0]}
HAVE2=$(find "${MEM_DIR}" -maxdepth 1 -name '*.pkl' | wc -l)
echo "[lme/${TIER}] DONE rc=${ADD_RC} have=${HAVE2}/${EXPECTED}"

# Cleanup server
kill -9 "${SERVER_PID}" 2>/dev/null || true
exit ${ADD_RC}
