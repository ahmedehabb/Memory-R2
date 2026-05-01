#!/bin/bash
# =============================================================================
# setup_judge_env.sh
#
# Pulls the official vllm/vllm-openai Docker image as an Apptainer SIF file.
# This is the recommended way to run gpt-oss on HPC (no Docker available).
# The image bundles the correct CUDA 12.8+ libs — no conda env needed.
#
# Run once:
#   bash setup_judge_env.sh
# =============================================================================

SIF_PATH=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm-openai.sif

if [ -f "$SIF_PATH" ]; then
    echo "[setup] SIF already exists at $SIF_PATH — skipping pull."
    echo "[setup] Delete it and rerun to refresh: rm $SIF_PATH"
    exit 0
fi

echo "[setup] Pulling vllm/vllm-openai Docker image via Apptainer..."
echo "[setup] This may take a while (image is several GB)..."
apptainer pull "$SIF_PATH" docker://vllm/vllm-openai

echo "[setup] Done: $SIF_PATH"
echo ""
echo "Test with:"
echo "  bash vllm_server_test.sh"
