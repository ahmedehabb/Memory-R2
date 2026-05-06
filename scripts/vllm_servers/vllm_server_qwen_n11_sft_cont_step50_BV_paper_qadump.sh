#!/bin/bash
# Paper-eval judge server: SFT-RL cont step50 (BESTVAL).
# Loads the SFT-RL model in vLLM, served as "openai/gpt-oss-120b" (compat name)
# at an isolated rendezvous dir so the test_eval client uses ONLY this server.

# Source .env so HF_TOKEN is available regardless of caller's env
if [ -f /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/.env ]; then
    set -a; source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/.env; set +a
fi

VLLM_JUDGE_MODEL="${VLLM_JUDGE_MODEL:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/outputs/answer_agent_sft_hf/n11_sft_cont_step50_BESTVAL_223256}"
VLLM_TENSOR_PARALLEL=${VLLM_TENSOR_PARALLEL:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}

SIF_PATH=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif
RENDEZVOUS_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers_qwen_n11_sft_cont_step50_BV_paper_qadump
mkdir -p "$RENDEZVOUS_DIR"

SERVER_IDX=${SERVER_IDX:-0}
VLLM_PORT=${VLLM_PORT:-$((8150 + SERVER_IDX))}
TIKTOKEN_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tiktoken_encodings

export HF_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES

LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server_qwen_step50_BV_paper
mkdir -p "$LOG_DIR" "$TIKTOKEN_DIR"

for enc in o200k_base cl100k_base; do
    if [ ! -f "$TIKTOKEN_DIR/${enc}.tiktoken" ]; then
        echo "[server] Downloading tiktoken vocab: ${enc}..."
        wget -q -O "$TIKTOKEN_DIR/${enc}.tiktoken" \
            "https://openaipublic.blob.core.windows.net/encodings/${enc}.tiktoken"
    fi
done

echo "$(hostname):${VLLM_PORT}" > "$RENDEZVOUS_DIR/server_${SERVER_IDX}.txt"
echo "[step50_BV-judge $SERVER_IDX] Rendezvous written: $(hostname):${VLLM_PORT}"
echo "[step50_BV-judge $SERVER_IDX] Starting vLLM server for $VLLM_JUDGE_MODEL ..."

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
    --served-model-name "openai/gpt-oss-120b" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.40 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --max-cudagraph-capture-size 2048 \
    --max-num-batched-tokens 8192 \
    --stream-interval 20 \
    $EXTRA_FLAGS \
    2>&1 | tee "$LOG_DIR/server_${SERVER_IDX}.log"

rm -f "$RENDEZVOUS_DIR/server_${SERVER_IDX}.txt"
