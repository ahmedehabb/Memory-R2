#!/usr/bin/env bash
# Re-test chain A — single-pipeline test_eval with val_kwargs.n=1 + REMA_DUMP_QA=1.
# Runs on one H200×4 alloc. Each row → ~25 min.
set -uo pipefail

REPO=<repo>
LOG_DIR=$REPO/logs/${SLURM_JOB_ID:-3989154}
mkdir -p $LOG_DIR
CHAIN_LOG=$LOG_DIR/re_test_chainA_$(date +%Y%m%d_%H%M%S).log
echo "[chain-A] start $(date -Is)" | tee "$CHAIN_LOG"

export REMA_DUMP_QA=1
export VAL_KWARGS_N=1

# Row 4: vl854fhl champion 7B 0.3
run_test() {
    local TAG=$1
    local CKPT=$2
    local TURNS=${3:-6}
    echo "[chain-A] === $TAG ===" | tee -a "$CHAIN_LOG"
    export RUN_TAG=$TAG
    export MODEL_PATH_OVERRIDE=$CKPT
    export MAX_NUM_TURNS=$TURNS
    export SERVER_WAIT_TIMEOUT=900
    bash $REPO/scripts/vllm_clients/vllm_client_test_eval.sh \
        > $LOG_DIR/${TAG}.log 2>&1
    rc=$?
    echo "[chain-A] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
}

run_test test_re_vl854fhl_champion_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

run_test test_re_xp2zzxm1_grpo_topk80_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_curr_32sess_inner0_topk80_pure_h200_r7_j3972431__20260419_220229_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

run_test test_re_ao1of33o_turns8_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_n8_turns8_32sess_retry_from8s__20260426_190917_8turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    8

run_test test_re_x134wabh_3b_32sess_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_p7_3b_direct_8_to_32__20260427_125613_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

echo "[chain-A] done $(date -Is)" | tee -a "$CHAIN_LOG"
