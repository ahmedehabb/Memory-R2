#!/bin/bash
# Launch vLLM server for 32sess on port 8001 (job 3968105 GPU 0)
SIF="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif"
HF_HOME="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home"
TIKTOKEN="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home/../tiktoken_encodings"
LOG="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server/server_32sess.log"
CKPT="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
PORT=${PORT:-8001}
CUDA_DEVICE=${CUDA_DEVICE:-0}

mkdir -p "$(dirname $LOG)"
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
echo "[$(date)] Starting 32sess server on GPU $CUDA_DEVICE port $PORT"
echo "Checkpoint: $CKPT" | tee "$LOG"

apptainer run --nv \
    --env HF_HOME="$HF_HOME" \
    --env CC=gcc --env CXX=g++ \
    --env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$TIKTOKEN:$TIKTOKEN" \
    --bind "$(dirname $CKPT):$(dirname $CKPT)" \
    "$SIF" \
    --model "$CKPT" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --served-model-name "32sess" \
    --max-num-seqs 64 \
    2>&1 | tee -a "$LOG"
