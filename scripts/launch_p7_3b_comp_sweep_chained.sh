#!/usr/bin/env bash
# Chained 3B compression sweep — runs comp=0.3 → 0.05 → 0.5 sequentially on the
# allocation. Each training: 8-sess, 10 epochs, ~1.5h on H200x4. Total ~4.5h.
set -uo pipefail

REPO_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public
LOG_BASE=${LOG_BASE:-$REPO_DIR/logs/3989152}
mkdir -p "$LOG_BASE"

CHAIN_LOG=$LOG_BASE/p7_3b_comp_sweep_chain_$(date +%Y%m%d_%H%M%S).log
echo "[chain] $(date -Is) starting 3B comp sweep" | tee "$CHAIN_LOG"

# Shared rendezvous to gpt-oss-120b on hkn1970:8107 (already registered in vllm_servers/)
export RENDEZVOUS_DIR=$REPO_DIR/vllm_servers

for COMP in 0.3 0.05 0.5; do
    RUN_TAG="p7_3b_comp${COMP/./}_8sess_h200_$(date +%H%M%S)"
    echo "[chain] $(date -Is) launching COMP=$COMP RUN_TAG=$RUN_TAG" | tee -a "$CHAIN_LOG"

    cd "$REPO_DIR"

    SKIP_NODE_CHECK=1 \
    RENDEZVOUS_DIR=$RENDEZVOUS_DIR \
    RUN_TAG=$RUN_TAG \
    CURRENT_MODEL_PATH_OVERRIDE='Qwen/Qwen2.5-3B-Instruct' \
    MAX_NUM_TURNS=6 \
    STAGES_OVERRIDE=8 \
    EPOCHS_PER_STAGE_OVERRIDE=10 \
    QA_TOP_K_PER_STAGE_OVERRIDE=30 \
    INNER_GPRO_FRAC=0.5 \
    COMPRESSION_PENALTY=$COMP \
    bash $REPO_DIR/scripts/vllm_clients/vllm_client_standalone.sh \
        > "$LOG_BASE/${RUN_TAG}.log" 2>&1
    rc=$?
    echo "[chain] $(date -Is) COMP=$COMP exited rc=$rc" | tee -a "$CHAIN_LOG"
    [ $rc -ne 0 ] && echo "[chain] WARNING: $RUN_TAG failed; continuing to next COMP" | tee -a "$CHAIN_LOG"
done

echo "[chain] $(date -Is) sweep complete (3 trainings done)" | tee -a "$CHAIN_LOG"
