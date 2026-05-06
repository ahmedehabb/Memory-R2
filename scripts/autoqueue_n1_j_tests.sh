#!/bin/bash
# Auto-queue next N1 J tests for a given SLURM job.
# Usage:
#   nohup bash scripts/autoqueue_n1_j_tests.sh <JOB_ID> <WAIT_PID> <queue_file> &
# queue_file format (TSV): ckpt_path<TAB>run_tag<TAB>max_num_turns<TAB>qa_dump_run_name
set -e

JID=$1
WAIT_PID=${2:-0}
QUEUE=$3
LOGDIR=logs/${JID}
mkdir -p $LOGDIR

echo "[autoqueue] jid=$JID wait_pid=$WAIT_PID queue=$QUEUE" | tee -a $LOGDIR/autoqueue.log

if [ "$WAIT_PID" != "0" ] && [ -e /proc/$WAIT_PID ]; then
  echo "[autoqueue] waiting for PID $WAIT_PID to exit..." | tee -a $LOGDIR/autoqueue.log
  while [ -e /proc/$WAIT_PID ]; do sleep 30; done
  echo "[autoqueue] PID $WAIT_PID exited" | tee -a $LOGDIR/autoqueue.log
fi

# After the initial wait, free GPU memory (kill any TP workers / stale vllm)
srun --jobid=$JID --overlap -N1 -n1 bash -c '
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  kill -9 $pid 2>/dev/null
done
sleep 5
' 2>/dev/null || true

while IFS=$'\t' read -r CKPT RUN_TAG MAX_TURNS RUN_NAME; do
  # skip comment/blank
  [[ -z "$CKPT" || "$CKPT" =~ ^# ]] && continue
  # Allow HF-hub ids (e.g. "Qwen/Qwen2.5-7B-Instruct") — only check dir existence for local paths
  if [[ "$CKPT" =~ ^/ || "$CKPT" =~ ^checkpoints/ ]]; then
    [ -d "$CKPT" ] || { echo "[autoqueue] skip $RUN_TAG — local ckpt missing: $CKPT" | tee -a $LOGDIR/autoqueue.log; continue; }
  else
    echo "[autoqueue] using HF hub id: $CKPT" | tee -a $LOGDIR/autoqueue.log
  fi

  echo "[autoqueue] $(date +%F' '%T) starting $RUN_TAG" | tee -a $LOGDIR/autoqueue.log
  # Pre-clean any stale GPU-holding process
  srun --jobid=$JID --overlap -N1 -n1 bash -c '
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $pid 2>/dev/null; done
    sleep 3
  ' 2>/dev/null || true

  VAL_KWARGS_N=1 MAX_NUM_TURNS=$MAX_TURNS REMA_DUMP_QA=1 \
  REMA_QA_DUMP_DIR=<repo>/qa_dumps \
  REMA_RUN_NAME=$RUN_NAME \
  MODEL_PATH_OVERRIDE=$CKPT RUN_TAG=$RUN_TAG \
    srun --jobid=$JID --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_test_eval.sh \
      > $LOGDIR/${RUN_TAG}_launch.log 2>&1 < /dev/null

  RC=$?
  echo "[autoqueue] $(date +%F' '%T) finished $RUN_TAG (rc=$RC)" | tee -a $LOGDIR/autoqueue.log
done < $QUEUE

echo "[autoqueue] $(date +%F' '%T) queue $QUEUE completed" | tee -a $LOGDIR/autoqueue.log
