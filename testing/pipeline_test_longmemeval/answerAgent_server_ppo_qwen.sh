#!/bin/bash

source ~/.bashrc
conda activate memoryr1

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=8000

# memAgent grpo
MODEL_NAME="ppo"
MODEL_PATH="/hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-split118-qwen-prompt-em-answer-reward-max-token-2048-Qwen2.5-7B-Instruct-0726-1/hf_converted/global_step_120" 
# Run grpo memAgent server at background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > answerAgent_ppo_qwen_server.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME"
echo "Parallel workers: $PARALLEL_NUM"
echo "Server running on $HOST:$PORT"
