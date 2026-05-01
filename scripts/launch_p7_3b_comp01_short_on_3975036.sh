#!/usr/bin/env bash
set -euo pipefail

JOB_ID=3975036
REPO_DIR="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public"
WATCH_LOG="${REPO_DIR}/logs/3975036/p7_3b_comp01_short_h200_watcher.log"
LAUNCH_LOG="${REPO_DIR}/logs/3975036/p7_3b_comp01_short_h200_launch.log"

mkdir -p "${REPO_DIR}/logs/3975036"
echo "[watcher] $(date -Is) waiting for job ${JOB_ID} to become RUNNING" >> "${WATCH_LOG}"

while true; do
  state="$(squeue -h -j "${JOB_ID}" -o "%T" 2>/dev/null | head -n1 || true)"
  if [[ "${state}" == "RUNNING" ]]; then
    echo "[watcher] $(date -Is) job ${JOB_ID} is RUNNING; launching short comp0.1 run" >> "${WATCH_LOG}"
    srun --jobid="${JOB_ID}" --overlap -N1 -n1 bash -lc "
      cd '${REPO_DIR}' &&
      export SKIP_NODE_CHECK=1 &&
      export RENDEZVOUS_DIR='${REPO_DIR}/vllm_servers_h100_shared' &&
      export RUN_TAG='p7_3b_comp01_short_h200' &&
      export CURRENT_MODEL_PATH_OVERRIDE='Qwen/Qwen2.5-3B-Instruct' &&
      export MAX_NUM_TURNS=6 &&
      export STAGES_OVERRIDE=8 &&
      export EPOCHS_PER_STAGE_OVERRIDE=5 &&
      export QA_TOP_K_PER_STAGE_OVERRIDE=30 &&
      export INNER_GPRO_FRAC=0.5 &&
      export COMPRESSION_PENALTY=0.1 &&
      bash scripts/vllm_clients/vllm_client_standalone.sh
    " >> "${LAUNCH_LOG}" 2>&1
    echo "[watcher] $(date -Is) launch command exited with code $?" >> "${WATCH_LOG}"
    break
  fi
  if [[ -z "${state}" ]]; then
    echo "[watcher] $(date -Is) job ${JOB_ID} not found in queue; exiting watcher" >> "${WATCH_LOG}"
    break
  fi
  sleep 30
done

