#!/bin/bash

source ~/.bashrc
conda activate memoryr1

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=8000

# memAgent grpo
MODEL_NAME="ppo"
# MODEL_PATH="/home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/memory_manager/global_step_15" 
MODEL_PATH="/hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-memAgent-multi-evidence-EM-score-copy-Llama-3.1-8B-Instruct-0908/hf_converted/global_step_10" 
# Run grpo memAgent server at background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > memAgent_ppo_llama_server.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME"
echo "Parallel workers: $PARALLEL_NUM"
echo "Server running on $HOST:$PORT"
