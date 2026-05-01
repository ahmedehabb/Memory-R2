#!/bin/bash

source ~/.bashrc
conda activate mem0

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=4399

MODEL_NAME="sft llama answerAgent"
MODEL_PATH="/hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/sft-split118-Llama-3.1-8B-Instruct-0925-1/merged_model/global_step_50"


nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > answerAgent_sft_llama_server.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME" 