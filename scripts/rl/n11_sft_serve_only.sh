#!/usr/bin/env bash
# Server-only: host an SFT-answer-agent vLLM server on this node, register the
# rendezvous file, and stay alive until killed. Pairs with n11_sft_eval_only.sh
# running on a DIFFERENT GPU allocation.
#
# Required env: SFT_HF_PATH, SFT_LABEL
# Optional env: VLLM_SERVER_GPU_MEM_UTIL (default 0.85 since this node is server-only)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SFT_HF_PATH="${SFT_HF_PATH:?SFT_HF_PATH=<abs path> required}"
SFT_LABEL="${SFT_LABEL:?SFT_LABEL=<short label> required}"

ISO_DIR="$REPO_ROOT/vllm_servers_qwen_${SFT_LABEL}"
mkdir -p "$ISO_DIR"
rm -f "$ISO_DIR"/server_*.txt

HASH_PORT=$(echo "$SFT_LABEL" | cksum | awk '{print 8100 + ($1 % 90)}')

export VLLM_JUDGE_MODEL="$SFT_HF_PATH"
export VLLM_PORT="$HASH_PORT"
export VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-4}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

ORIG_RV=<repo>/vllm_servers_qwen
TMP_SERVER="$REPO_ROOT/vllm_server_qwen_${SFT_LABEL}.sh"
sed -e "s|$ORIG_RV|$ISO_DIR|g" \
    -e "s|--gpu-memory-utilization 0.85|--gpu-memory-utilization ${VLLM_SERVER_GPU_MEM_UTIL:-0.85}|" \
    "$REPO_ROOT/vllm_server_qwen.sh" > "$TMP_SERVER"
chmod +x "$TMP_SERVER"

echo "[n11_serve_only] $SFT_LABEL :: serving $SFT_HF_PATH on port $HASH_PORT"
echo "[n11_serve_only] iso rendezvous = $ISO_DIR"
echo "[n11_serve_only] gpu_mem_util  = ${VLLM_SERVER_GPU_MEM_UTIL:-0.85}"
exec bash "$TMP_SERVER"
