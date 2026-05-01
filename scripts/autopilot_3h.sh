#!/usr/bin/env bash
set -euo pipefail

# ReMA autopilot:
# - checks active allocations/runs
# - launches next queued experiments when a training slot becomes idle
# - logs status every cycle
#
# Default interval: 3 hours (10800s)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-10800}"
LOG_DIR="${ROOT_DIR}/logs"
AUTO_LOG="${LOG_DIR}/autopilot_3h.log"
STATE_FILE="${LOG_DIR}/autopilot_3h.state"
LOCK_FILE="${LOG_DIR}/autopilot_3h.lock"

mkdir -p "${LOG_DIR}"

if [[ -f "${LOCK_FILE}" ]]; then
  old_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[autopilot] already running with pid=${old_pid}" | tee -a "${AUTO_LOG}"
    exit 0
  fi
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

if [[ ! -f "${STATE_FILE}" ]]; then
  cat > "${STATE_FILE}" <<'EOF'
NEXT_QUEUE_INDEX=0
EOF
fi

# Queue format:
# name|job_id|launch_command
QUEUE_ITEMS=(
  "p3_comp01_8sess|3972430|srun --jobid=3972430 --overlap bash -lc 'export RENDEZVOUS_DIR=${ROOT_DIR}/vllm_servers_h100_shared; export RUN_TAG=curr_8sess_p3_comp01_8sess_fix_answeragent; export COMPRESSION_PENALTY=0.1; export MAX_NUM_TURNS=4; export STAGES_OVERRIDE=8; export EPOCHS_PER_STAGE_OVERRIDE=10; export QA_TOP_K_PER_STAGE_OVERRIDE=30; bash scripts/vllm_clients/vllm_client_standalone.sh'"
  "p8_memops_topk5_8sess|3973071|srun --jobid=3973071 --overlap bash -lc 'export RENDEZVOUS_DIR=${ROOT_DIR}/vllm_servers_h100_shared; export RUN_TAG=curr_8sess_p8_memops_topk5_8sess_fix_answeragent; export COMPRESSION_PENALTY=0.2; export TOP_K_MEMORIES=5; export MAX_NUM_TURNS=4; export STAGES_OVERRIDE=8; export EPOCHS_PER_STAGE_OVERRIDE=10; export QA_TOP_K_PER_STAGE_OVERRIDE=30; bash scripts/vllm_clients/vllm_client_standalone.sh'"
)

load_state() {
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
}

save_state() {
  cat > "${STATE_FILE}" <<EOF
NEXT_QUEUE_INDEX=${NEXT_QUEUE_INDEX}
EOF
}

is_alloc_running() {
  local job_id="$1"
  local state
  state="$(squeue -h -j "${job_id}" -o "%T" 2>/dev/null | head -n 1 || true)"
  [[ "${state}" == "RUNNING" ]]
}

job_has_active_trainer() {
  local job_id="$1"
  local out
  out="$(srun --jobid="${job_id}" --overlap -N1 -n1 bash -lc \
    "ps -eo cmd | rg -n 'python -m verl.rema_trainer.main_ppo|python -m verl.rema_separated_trainer.main_ppo' || true" \
    2>/dev/null || true)"
  [[ -n "${out}" ]]
}

log_snapshot() {
  local now
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  {
    echo
    echo "===== [${now}] AUTOPILOT CYCLE ====="
    squeue -u "${USER}" || true
    for jid in 3972430 3973071 3975990 3975991; do
      if is_alloc_running "${jid}"; then
        echo "[alloc ${jid}] RUNNING"
      else
        echo "[alloc ${jid}] NOT_RUNNING"
      fi
    done
  } >> "${AUTO_LOG}"
}

try_launch_next() {
  load_state

  if [[ "${NEXT_QUEUE_INDEX}" -ge "${#QUEUE_ITEMS[@]}" ]]; then
    echo "[autopilot] queue complete (NEXT_QUEUE_INDEX=${NEXT_QUEUE_INDEX})" >> "${AUTO_LOG}"
    return 0
  fi

  local item="${QUEUE_ITEMS[${NEXT_QUEUE_INDEX}]}"
  IFS='|' read -r name job_id launch_cmd <<< "${item}"

  if ! is_alloc_running "${job_id}"; then
    echo "[autopilot] ${name}: allocation ${job_id} not running; deferring" >> "${AUTO_LOG}"
    return 0
  fi

  if job_has_active_trainer "${job_id}"; then
    echo "[autopilot] ${name}: job ${job_id} busy with active trainer; deferring" >> "${AUTO_LOG}"
    return 0
  fi

  local launch_log="${LOG_DIR}/${name}_autolaunch.log"
  echo "[autopilot] launching ${name} on allocation ${job_id}" | tee -a "${AUTO_LOG}"
  nohup bash -lc "${launch_cmd}" >> "${launch_log}" 2>&1 &
  local pid=$!
  echo "[autopilot] launched ${name} (nohup pid=${pid}) log=${launch_log}" | tee -a "${AUTO_LOG}"

  NEXT_QUEUE_INDEX=$((NEXT_QUEUE_INDEX + 1))
  save_state
}

echo "[autopilot] started pid=$$ interval=${INTERVAL_SECONDS}s" | tee -a "${AUTO_LOG}"

while true; do
  log_snapshot
  try_launch_next
  sleep "${INTERVAL_SECONDS}"
done

