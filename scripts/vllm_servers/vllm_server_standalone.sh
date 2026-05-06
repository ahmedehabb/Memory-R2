#!/bin/bash
# Run on the SERVER node:
#   srun --jobid=<server_job_id> bash vllm_server_standalone.sh

VLLM_JUDGE_MODEL=${VLLM_JUDGE_MODEL:-"openai/gpt-oss-120b"}
VLLM_TENSOR_PARALLEL=${VLLM_TENSOR_PARALLEL:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}

SIF_PATH=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif
RENDEZVOUS_DIR=${RENDEZVOUS_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers}
mkdir -p "$RENDEZVOUS_DIR"

# Auto-detect next available server index from existing rendezvous files
SERVER_IDX=${SERVER_IDX:-$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null | wc -l)}
# Auto-assign port based on index (8000, 8001, 8002, ...)
VLLM_PORT=${VLLM_PORT:-$((8000 + SERVER_IDX))}
TIKTOKEN_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tiktoken_encodings

export HF_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES

LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server
mkdir -p "$LOG_DIR" "$TIKTOKEN_DIR"

# Download tiktoken vocab if missing
for enc in o200k_base cl100k_base; do
    if [ ! -f "$TIKTOKEN_DIR/${enc}.tiktoken" ]; then
        echo "[server] Downloading tiktoken vocab: ${enc}..."
        wget -q -O "$TIKTOKEN_DIR/${enc}.tiktoken" \
            "https://openaipublic.blob.core.windows.net/encodings/${enc}.tiktoken"
    fi
done

# Write hostname:port so the client job can find this server
echo "$(hostname):${VLLM_PORT}" > "$RENDEZVOUS_DIR/server_${SERVER_IDX}.txt"
echo "[server $SERVER_IDX] Rendezvous written: $(hostname):${VLLM_PORT}"
echo "[server $SERVER_IDX] Starting vLLM server for $VLLM_JUDGE_MODEL ..."

COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "9.0")
if [ "$COMPUTE_CAPABILITY" = "10.0" ]; then
    export APPTAINERENV_VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
    EXTRA_FLAGS="--kv-cache-dtype fp8"
else
    EXTRA_FLAGS=""
fi

apptainer run --nv \
    --env HF_TOKEN="$HF_TOKEN" \
    --env HF_HOME="$HF_HOME" \
    --env CC=gcc \
    --env CXX=g++ \
    --env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_DIR" \
    --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    --bind "$HF_HOME":"$HF_HOME" \
    --bind "$TIKTOKEN_DIR":"$TIKTOKEN_DIR" \
    "$SIF_PATH" \
    --model "$VLLM_JUDGE_MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --max-cudagraph-capture-size 2048 \
    --max-num-batched-tokens 8192 \
    --stream-interval 20 \
    $EXTRA_FLAGS \
    2>&1 | tee "$LOG_DIR/server_${SERVER_IDX}.log"

# Clean up rendezvous file on exit
rm -f "$RENDEZVOUS_DIR/server_${SERVER_IDX}.txt"
