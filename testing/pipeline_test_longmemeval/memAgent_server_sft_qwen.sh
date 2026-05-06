#!/bin/bash

source ~/.bashrc
conda activate mem0

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=8000

MODEL_NAME="sft qwen 7b memAgent"
MODEL_PATH="<scratch>/<user>-mem0rl/sft-memAgent-multi-evidence-Qwen-7B-Instruct-1001/merged_model/global_step_60"


nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > memAgent_server_sft_qwen.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME" 