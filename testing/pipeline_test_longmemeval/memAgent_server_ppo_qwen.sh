#!/bin/bash

source ~/.bashrc
conda activate memoryr1

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=8000

# memAgent grpo
MODEL_NAME="ppo"
MODEL_PATH="<scratch>/<user>-mem0rl/ppo-memAgent-Qwen2.5-7B-Instruct-multi-evidence-EM-reward-copy-0918/hf_converted/global_step_55" 
# Run grpo memAgent server at background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > memAgent_ppo_server_qwen.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME"
echo "Parallel workers: $PARALLEL_NUM"
echo "Server running on $HOST:$PORT"
