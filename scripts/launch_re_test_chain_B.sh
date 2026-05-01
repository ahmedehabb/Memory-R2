#!/usr/bin/env bash
# Re-test chain B — separated_params test_eval with val_kwargs.n=1 + REMA_DUMP_QA=1.
# Runs on one H200×4 alloc.
set -uo pipefail

REPO=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public
LOG_DIR=$REPO/logs/${SLURM_JOB_ID:-3989153}
mkdir -p $LOG_DIR
CHAIN_LOG=$LOG_DIR/re_test_chainB_$(date +%Y%m%d_%H%M%S).log
echo "[chain-B] start $(date -Is)" | tee "$CHAIN_LOG"

export REMA_DUMP_QA=1
export VAL_KWARGS_N=1

run_test_sep() {
    local TAG=$1
    local META=$2
    local REASONING=$3
    local START=${4:-meta_thinking}
    local TURNS=${5:-4}
    echo "[chain-B] === $TAG ===" | tee -a "$CHAIN_LOG"
    export RUN_TAG=$TAG
    export MODEL_PATH_META=$META
    export MODEL_PATH_REASONING=$REASONING
    export START_AGENT=$START
    export MAX_NUM_TURNS=$TURNS
    export SERVER_WAIT_TIMEOUT=900
    bash $REPO/scripts/vllm_clients/vllm_client_test_eval_separated.sh \
        > $LOG_DIR/${TAG}.log 2>&1
    rc=$?
    echo "[chain-B] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
}

run_test() {
    local TAG=$1
    local CKPT=$2
    local TURNS=${3:-6}
    echo "[chain-B] === $TAG ===" | tee -a "$CHAIN_LOG"
    export RUN_TAG=$TAG
    export MODEL_PATH_OVERRIDE=$CKPT
    export MAX_NUM_TURNS=$TURNS
    export SERVER_WAIT_TIMEOUT=900
    bash $REPO/scripts/vllm_clients/vllm_client_test_eval.sh \
        > $LOG_DIR/${TAG}.log 2>&1
    rc=$?
    echo "[chain-B] $TAG rc=$rc $(date -Is)" | tee -a "$CHAIN_LOG"
}

# Row 18: P8 BT (only mem-mgr) — okh7h8or
P8BT=$REPO/checkpoints/rema-curriculum-v1/32sess_p8_base_trained_switch200_startreasoning_turns4_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5/global_step_5
run_test_sep test_re_p8_base_trained_n1 \
    $P8BT/meta_thinking/hf_fixed \
    $P8BT/reasoning/hf_fixed \
    reasoning 4

# Row 19: P8 TB v8 (only fact-extr) — i5kcmid7
P8TB=$REPO/checkpoints/rema-curriculum-v1/32sess_p8_trained_base_v8_switch200_startmeta_thinking_turns4_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5/global_step_5
run_test_sep test_re_p8_trained_base_v8_n1 \
    $P8TB/meta_thinking/hf_fixed \
    $P8TB/reasoning/hf_fixed \
    meta_thinking 4

# Row 17: ly7e63wd N5 sep 32sess
LYSEP=$REPO/checkpoints/rema-curriculum-v1/32sess_separated_n5_params_switch10_startmeta_thinking_turns4_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5/global_step_5
run_test_sep test_re_ly7e63wd_n5_sep_n1 \
    $LYSEP/meta_thinking/hf_fixed \
    $LYSEP/reasoning/hf_fixed \
    meta_thinking 4

# Row 13: 002h8p51 turns=4 single-pipeline
N4=$(ls -d $REPO/checkpoints/rema-curriculum-v1/curr_32sess_n8_turns4_32sess__*/global_step_5/hf_fixed 2>/dev/null | head -1)
[ -n "$N4" ] && run_test test_re_002h8p51_turns4_n1 "$N4" 4

# Row 15: ljsnigle turns=10 single-pipeline
N10=$(ls -d $REPO/checkpoints/rema-curriculum-v1/curr_32sess_*turns10*/global_step_5/hf_fixed 2>/dev/null | head -1)
[ -n "$N10" ] && run_test test_re_ljsnigle_turns10_n1 "$N10" 10

echo "[chain-B] done $(date -Is)" | tee -a "$CHAIN_LOG"
