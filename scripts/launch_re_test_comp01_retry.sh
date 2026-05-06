#!/usr/bin/env bash
# Re-test 7B comp=0.1 RETRY (Item I) — proves λ=0.1 collapse is seed-dependent.
# val_kwargs.n=1 + REMA_DUMP_QA=1 → 1085 unique QAs.
set -uo pipefail

REPO=<repo>
LOG_DIR=$REPO/logs/${SLURM_JOB_ID:-3994517}
mkdir -p $LOG_DIR
CHAIN_LOG=$LOG_DIR/re_test_comp01_retry_$(date +%Y%m%d_%H%M%S).log
echo "[retry-comp01] start $(date -Is)" | tee "$CHAIN_LOG"

export REMA_DUMP_QA=1
export VAL_KWARGS_N=1

TAG=test_re_comp01_RETRY_7B_n1
CKPT=$REPO/checkpoints/rema-curriculum-v1/curr_32sess_n7_comp01_32sess_RETRY_20260429_080318__20260429_080318_6turns_2ppo_Kl0.001_persession_0.1addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed

echo "[retry-comp01] === $TAG ===" | tee -a "$CHAIN_LOG"
export RUN_TAG=$TAG
export MODEL_PATH_OVERRIDE=$CKPT
export MAX_NUM_TURNS=6
export SERVER_WAIT_TIMEOUT=900

bash $REPO/scripts/vllm_clients/vllm_client_test_eval.sh \
    > $LOG_DIR/${TAG}.log 2>&1
rc=$?
echo "[retry-comp01] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
echo "[retry-comp01] done $(date -Is)" | tee -a "$CHAIN_LOG"
