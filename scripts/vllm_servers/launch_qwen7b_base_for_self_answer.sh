#!/bin/bash
# Launch vLLM server hosting base Qwen-2.5-7B-Instruct for the "self-answer" eval
# (untrained memory manager + untrained Qwen-7B as answer agent).
# Writes a rendezvous file at vllm_servers/server_qwen7b_self.txt.

set -euo pipefail

SIF="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif"
HF_HOME="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home"
TIKTOKEN="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home/../tiktoken_encodings"
RENDEZVOUS_DIR="${RENDEZVOUS_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers_qwen7b_self}"
LOG_DIR="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server"
mkdir -p "$RENDEZVOUS_DIR" "$LOG_DIR"

CKPT="Qwen/Qwen2.5-7B-Instruct"
PORT=${PORT:-8007}
HOST=$(hostname -f)
LOG="$LOG_DIR/server_qwen7b_self_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "[$(date)] Starting Qwen-7B-Instruct vLLM server on host=$HOST port=$PORT cuda=$CUDA_VISIBLE_DEVICES" | tee "$LOG"

# Write rendezvous AFTER server is up — done in a watcher
( while true; do
    if curl -sf --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "${HOST}:${PORT}" > "$RENDEZVOUS_DIR/server_0.txt"
        echo "[rendezvous] wrote $RENDEZVOUS_DIR/server_0.txt -> ${HOST}:${PORT}" | tee -a "$LOG"
        break
    fi
    sleep 5
done ) &

apptainer run --nv \
    --env HF_HOME="$HF_HOME" \
    --env CC=gcc --env CXX=g++ \
    --env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$TIKTOKEN:$TIKTOKEN" \
    "$SIF" \
    --model "$CKPT" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "qwen7b-self" \
    --max-num-seqs 64 \
    2>&1 | tee -a "$LOG"
