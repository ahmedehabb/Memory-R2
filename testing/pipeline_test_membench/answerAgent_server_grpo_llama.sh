#!/bin/bash

source ~/.bashrc
conda activate memoryr1

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=8000

# memAgent grpo
MODEL_NAME="grpo"
MODEL_PATH="/home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_30" 
# Run grpo memAgent server at background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > answerAgent_grpo_llama_server.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME"
echo "Parallel workers: $PARALLEL_NUM"
echo "Server running on $HOST:$PORT"
