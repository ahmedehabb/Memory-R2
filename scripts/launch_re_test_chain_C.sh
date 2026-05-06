#!/usr/bin/env bash
# Phase C — re-test 7B compression sweep (rows 1, 2, 3, 5) + 3B compression sweep (rows 6, 7, 8, 9, 10) + curriculum row 12
# val_kwargs.n=1 + REMA_DUMP_QA=1 → 1085 unique QAs, no dedup needed.
set -uo pipefail

REPO=<repo>
LOG_DIR=$REPO/logs/${SLURM_JOB_ID:-3989152}
mkdir -p $LOG_DIR
CHAIN_LOG=$LOG_DIR/re_test_chainC_$(date +%Y%m%d_%H%M%S).log
echo "[chain-C] start $(date -Is)" | tee "$CHAIN_LOG"

export REMA_DUMP_QA=1
export VAL_KWARGS_N=1

run_test() {
    local TAG=$1
    local CKPT=$2
    local TURNS=${3:-6}
    echo "[chain-C] === $TAG ===" | tee -a "$CHAIN_LOG"
    export RUN_TAG=$TAG
    export MODEL_PATH_OVERRIDE=$CKPT
    export MAX_NUM_TURNS=$TURNS
    export SERVER_WAIT_TIMEOUT=900
    bash $REPO/scripts/vllm_clients/vllm_client_test_eval.sh \
        > $LOG_DIR/${TAG}.log 2>&1
    rc=$?
    echo "[chain-C] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
}

# ── 7B compression sweep ────────────────────────────────────────
# Row 1: 7B comp=0 (24mm5co7) — 32sess
run_test test_re_24mm5co7_7B_comp0_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_3985666_6turns_2ppo_Kl0.001_persession_0addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

# Row 2: 7B comp=0.05 (5k0nxeva)
run_test test_re_5k0nxeva_7B_comp005_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_n7_comp005_32sess__20260426_164509_6turns_2ppo_Kl0.001_persession_0.05addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

# Row 3: 7B comp=0.1 (if84og42 collapsed run)
run_test test_re_if84og42_7B_comp01_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_n7_comp01_32sess__20260426_130641_6turns_2ppo_Kl0.001_persession_0.1addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

# Row 5: 7B comp=0.5 (ae563vbc)
run_test test_re_ae563vbc_7B_comp05_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_32sess_3985667_6turns_2ppo_Kl0.001_persession_0.5addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed \
    6

# ── 3B compression sweep ───────────────────────────────────────
# Row 6: 3B comp=0 (zoesecfg)
run_test test_re_zoesecfg_3B_comp0_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_comp00_3972430__20260418_160726_6turns_2ppo_Kl0.001_persession_0.0addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    6

# Row 7: 3B comp=0.05
run_test test_re_3B_comp005_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_comp005_8sess_h200_001943__20260429_001943_6turns_2ppo_Kl0.001_persession_0.05addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    6

# Row 8: 3B comp=0.1 (q3gaqba4)
run_test test_re_q3gaqba4_3B_comp01_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_comp01_full_h200_j3975034__20260419_114306_6turns_2ppo_Kl0.001_persession_0.1addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    6

# Row 9: 3B comp=0.3 (3B champion)
run_test test_re_3B_comp03_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_comp03_8sess_h200_195306__20260428_195306_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    6

# Row 10: 3B comp=0.5 (retry)
run_test test_re_3B_comp05_n1 \
    $REPO/checkpoints/rema-curriculum-v1/curr_8sess_p7_3b_comp05_retry_20260429_075721__20260429_075721_6turns_2ppo_Kl0.001_persession_0.5addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    6

echo "[chain-C] done $(date -Is)" | tee -a "$CHAIN_LOG"
