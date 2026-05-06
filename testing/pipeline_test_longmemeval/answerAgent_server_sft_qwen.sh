#!/bin/bash

source ~/.bashrc
conda activate mem0

PARALLEL_NUM=4  # Number of parallel workers
HOST="0.0.0.0"
PORT=4399

MODEL_NAME="sft qwen answerAgent"
MODEL_PATH="<scratch>/<user>-mem0rl/sft-split118-Qwen2.5-7B-Instruct-1001-1/merged_model/global_step_60"


nohup python3 -m vllm.entrypoints.openai.api_server \
    --host $HOST \
    --port $PORT \
    --model $MODEL_PATH \
    --tensor-parallel-size $PARALLEL_NUM \
    > answerAgent_sft_qwen_server.log 2>&1 &

echo "vllm server started at background (PID $!)"
echo "Model: $MODEL_NAME" 