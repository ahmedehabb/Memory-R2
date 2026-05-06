#!/bin/bash
CKPT="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/curr_32sess_32sess_continued_lowlr__20260412_012917_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
SIF="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif"
HF_HOME="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home"
LOG="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/vllm_server/rema_trained_server.log"
TIKTOKEN_DIR="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tiktoken_encodings"

mkdir -p "$(dirname $LOG)"
export CUDA_VISIBLE_DEVICES=1
echo "[$(date)] Starting ReMA trained model server on GPU 1, port 8011" | tee "$LOG"

apptainer run --nv \
    --env HF_HOME="$HF_HOME" \
    --env CC=gcc \
    --env CXX=g++ \
    --env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_DIR" \
    --bind "$HF_HOME":"$HF_HOME" \
    --bind "$TIKTOKEN_DIR":"$TIKTOKEN_DIR" \
    --bind "$(dirname $CKPT)":"$(dirname $CKPT)" \
    "$SIF" \
    --model "$CKPT" \
    --host 0.0.0.0 \
    --port 8011 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --served-model-name rema-best \
    --max-num-seqs 32 \
    2>&1 | tee -a "$LOG"
