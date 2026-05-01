#!/usr/bin/env bash
# Test_eval the 3 finished 3B λ-consistent ckpts (λ=0.0, 0.05, 0.1) sequentially.
# val_kwargs.n=1 + REMA_DUMP_QA=1 → 1085 unique QAs, no dedup.
set -uo pipefail

REPO=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public
LOG_DIR=$REPO/logs/${SLURM_JOB_ID:-4003879}
mkdir -p $LOG_DIR
CHAIN_LOG=$LOG_DIR/3b_consistent_test_chain_$(date +%Y%m%d_%H%M%S).log
echo "[3b-test-chain] start $(date -Is)" | tee "$CHAIN_LOG"

export REMA_DUMP_QA=1
export VAL_KWARGS_N=1

run_test() {
    local TAG=$1
    local CKPT=$2
    echo "[3b-test-chain] === $TAG ===" | tee -a "$CHAIN_LOG"
    export RUN_TAG=$TAG
    export MODEL_PATH_OVERRIDE=$CKPT
    export MAX_NUM_TURNS=6
    export SERVER_WAIT_TIMEOUT=900
    bash $REPO/scripts/vllm_clients/vllm_client_test_eval.sh \
        > $LOG_DIR/${TAG}.log 2>&1
    rc=$?
    echo "[3b-test-chain] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
}

# 3B λ=0.0 32sess (consistent)
run_test test_re_3b_lambda00_consistent_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_p7_3b_32sess_lambda00_consistent_20260430_010738__20260430_010739_6turns_2ppo_Kl0.001_persession_0.0addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed

# 3B λ=0.05 32sess (consistent)
run_test test_re_3b_lambda005_consistent_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_p7_3b_32sess_lambda005_consistent_20260430_010738__20260430_010739_6turns_2ppo_Kl0.001_persession_0.05addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed

# 3B λ=0.1 32sess (consistent)
run_test test_re_3b_lambda01_consistent_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_p7_3b_32sess_lambda01_consistent_20260430_010738__20260430_010739_6turns_2ppo_Kl0.001_persession_0.1addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed

echo "[3b-test-chain] done $(date -Is)" | tee -a "$CHAIN_LOG"
