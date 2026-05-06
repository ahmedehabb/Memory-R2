#!/usr/bin/env bash
# N11: on a single H100/H200 allocation, host the SFT-answer-agent vLLM server
# and run the LoCoMo test eval against the 32sess_champion_v2 mem-policy.
#
# Required env:
#   SFT_HF_PATH=<absolute path under outputs/answer_agent_sft_hf/>
#   SFT_LABEL=<short label e.g. n11_sft_step20>
# Optional env:
#   MEM_CKPT_PATH=<rema-curriculum-v1 ckpt>  default: 32sess_champion_v2 step5 hf_fixed
#   VLLM_TENSOR_PARALLEL=4
#
# Strategy: each call uses an ISOLATED rendezvous dir (vllm_servers_qwen_<LABEL>/)
# so multiple instances on different GPU nodes never see each other's servers.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SFT_HF_PATH="${SFT_HF_PATH:?SFT_HF_PATH=<abs path> required}"
SFT_LABEL="${SFT_LABEL:?SFT_LABEL=<short label> required}"
TS="$(date +%Y%m%d_%H%M%S)"

# Isolated rendezvous dir so concurrent runs don't cross-talk.
ISO_DIR="$REPO_ROOT/vllm_servers_qwen_${SFT_LABEL}"
mkdir -p "$ISO_DIR"
rm -f "$ISO_DIR"/server_*.txt

# Per-run port offset to avoid collision when 2 nodes happen to schedule on adjacent IPs.
HASH_PORT=$(echo "$SFT_LABEL" | cksum | awk '{print 8100 + ($1 % 90)}')

# Env for the server
export VLLM_JUDGE_MODEL="$SFT_HF_PATH"
export VLLM_PORT="$HASH_PORT"
export VLLM_TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-4}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# Backup the global rendezvous dir bind (server script writes to a hardcoded path)
SERVER_SCRIPT="$REPO_ROOT/vllm_server_qwen.sh"

# Start server in background — bind RENDEZVOUS_DIR to our isolated dir via env hack
ORIG_RV=<repo>/vllm_servers_qwen
SERVER_LOG="$REPO_ROOT/logs/n11_serve_eval_${SFT_LABEL}_${TS}.server.log"

echo "[n11_serve_eval] $SFT_LABEL :: serving $SFT_HF_PATH on port $HASH_PORT"
echo "[n11_serve_eval] iso rendezvous = $ISO_DIR"
echo "[n11_serve_eval] server log     = $SERVER_LOG"

# Patch-on-the-fly: copy server script to a temp location with replaced rendezvous path
# AND reduce gpu_memory_utilization from 0.85 -> 0.40 so the eval client (which loads
# its own Qwen2.5-7B memory pipeline on the same GPUs) has enough VRAM headroom.
TMP_SERVER="$REPO_ROOT/vllm_server_qwen_${SFT_LABEL}.sh"
sed -e "s|$ORIG_RV|$ISO_DIR|g" \
    -e "s|--gpu-memory-utilization 0.85|--gpu-memory-utilization ${VLLM_SERVER_GPU_MEM_UTIL:-0.40}|" \
    "$SERVER_SCRIPT" > "$TMP_SERVER"
chmod +x "$TMP_SERVER"

bash "$TMP_SERVER" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[n11_serve_eval] server PID=$SERVER_PID; waiting for readiness..."

# Wait up to 20 minutes for the server to register
for i in $(seq 1 240); do
  if ls "$ISO_DIR"/server_*.txt >/dev/null 2>&1; then
    # Check server actually responds
    URL=$(cat "$ISO_DIR"/server_0.txt 2>/dev/null)
    if [ -n "$URL" ]; then
      if curl -fsS "http://${URL}/v1/models" > /dev/null 2>&1; then
        echo "[n11_serve_eval] server ready at $URL after $((i*5))s"
        break
      fi
    fi
  fi
  sleep 5
done

if ! ls "$ISO_DIR"/server_*.txt >/dev/null 2>&1; then
  echo "[n11_serve_eval] ERROR: server never registered in $ISO_DIR; killing PID=$SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

# Run the eval against the now-running server
export JUDGE_RENDEZVOUS_DIR="$ISO_DIR"
export MODEL_PATH_OVERRIDE="${MEM_CKPT_PATH:-$REPO_ROOT/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed}"
export RUN_TAG="${SFT_LABEL}_eval_${TS}"

export EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.40}"
echo "[n11_serve_eval] starting test-eval RUN_TAG=$RUN_TAG (server_mem=${VLLM_SERVER_GPU_MEM_UTIL:-0.40} eval_mem=$EVAL_GPU_MEM_UTIL)"
bash "$REPO_ROOT/scripts/vllm_clients/vllm_client_test_eval_qwen.sh"
RC=$?
echo "[n11_serve_eval] test-eval rc=$RC for $RUN_TAG"

# Tear down server + tmp script
echo "[n11_serve_eval] killing server PID=$SERVER_PID"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
rm -f "$TMP_SERVER"
rm -f "$ISO_DIR"/server_*.txt

exit $RC
