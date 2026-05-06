#!/bin/bash
# N6: P8 component swap without training.
# Waits for the given SLURM job's GPUs to go idle, then fires the separated-trainer
# test_only pass with META vs REASONING model-path swap.
#
# Usage:
#   nohup bash scripts/autoqueue_n6.sh <JID> <RUN_TAG> <MODEL_META> <MODEL_REASONING> &
#
# All paths accepted both as local dirs (hf_fixed) or HF hub ids (e.g. Qwen/Qwen2.5-7B-Instruct).

set -u

JID=$1
RUN_TAG=$2
MODEL_META=$3
MODEL_REASONING=$4

LOGDIR=logs/${JID}
mkdir -p $LOGDIR

echo "[n6] jid=$JID tag=$RUN_TAG meta=$MODEL_META memory=$MODEL_REASONING" | tee -a $LOGDIR/autoqueue_n6.log

# Wait until all 4 GPUs drop below 30% util for 60s (node is idle)
STABLE=0
while true; do
  utils=$(srun --jobid=$JID --overlap -N1 -n1 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  max_util=$(echo "$utils" | sort -rn | head -1)
  if [ -z "$max_util" ]; then max_util=100; fi
  if [ "$max_util" -lt 30 ]; then
    STABLE=$((STABLE+1))
  else
    STABLE=0
  fi
  if [ $STABLE -ge 6 ]; then
    echo "[n6] $(date +%F' '%T) node idle confirmed (max_util=$max_util for 60s)" | tee -a $LOGDIR/autoqueue_n6.log
    break
  fi
  sleep 10
done

# Free any residual GPU memory
srun --jobid=$JID --overlap -N1 -n1 bash -c '
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do kill -9 $pid 2>/dev/null; done
sleep 5
' 2>/dev/null || true

# Fire N6 test_only — uses separated-trainer, QA-dump for J later
VAL_KWARGS_N=1 MAX_NUM_TURNS=6 REMA_DUMP_QA=1 \
REMA_QA_DUMP_DIR=<repo>/qa_dumps \
REMA_RUN_NAME=$RUN_TAG \
MODEL_PATH_META=$MODEL_META \
MODEL_PATH_REASONING=$MODEL_REASONING \
RUN_TAG=$RUN_TAG \
  srun --jobid=$JID --overlap -N1 -n1 bash scripts/vllm_clients/vllm_client_test_eval_separated.sh \
    > $LOGDIR/${RUN_TAG}_launch.log 2>&1 < /dev/null

echo "[n6] $(date +%F' '%T) finished $RUN_TAG (rc=$?)" | tee -a $LOGDIR/autoqueue_n6.log
