#!/bin/bash
# Launch vLLM server for base on port 8007
SIF="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif"
HF_HOME="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home"
TIKTOKEN="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home/../tiktoken_encodings"
LOG="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server/server_base.log"
CKPT="Qwen/Qwen2.5-7B-Instruct"
PORT=${PORT:-8007}
CUDA_DEVICE=${CUDA_DEVICE:-0}

mkdir -p "$(dirname $LOG)"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
echo "[$(date)] Starting base server on GPU $CUDA_DEVICE port $PORT"
echo "Checkpoint: $CKPT" | tee "$LOG"

apptainer run --nv \
    --env HF_HOME="$HF_HOME" \
    --env CC=gcc --env CXX=g++ \
    --env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$TIKTOKEN:$TIKTOKEN" \
    "$SIF" \
    --model "$CKPT" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "base" \
    --max-num-seqs 64 \
    2>&1 | tee -a "$LOG"
