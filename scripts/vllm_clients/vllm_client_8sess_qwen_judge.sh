#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# G10: 8-sess training using Qwen2.5-7B as reward judge instead of gpt-oss-120b
# Tests open-source reproducibility: does the method work without closed commercial models?
#
# Key differences vs standard 8-sess:
#   - RENDEZVOUS_DIR → vllm_servers_qwen/ (Qwen judge, not gpt-oss)
#   - VLLM_PORT=8100, VLLM_JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
#   - Everything else same as 8sess_turns6 champion config
#
# Comparison: champion 8-sess (gpt-oss judge) val=0.498, mfail=0.016
# Expected: similar or slightly lower — if within ~0.02, method is reproducible open-source
#
# Start vllm_server_qwen.sh on a separate H100 before running this.
# Usage: srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_8sess_qwen_judge.sh

export SKIP_NODE_CHECK=1

# Point to Qwen judge server instead of gpt-oss
export RENDEZVOUS_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers_qwen
export VLLM_PORT=8100
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Same config as 8sess_turns6 champion
export MAX_NUM_TURNS=6
export CLIP_RATIO=0.2
export CLIP_MODE=turn
export AGG_MODE=turn
export INNER_GPRO_FRAC=0.5
export QA_TOP_K_PER_STAGE_OVERRIDE=30
export STAGES_OVERRIDE=8
export EPOCHS_PER_STAGE_OVERRIDE=10
export COMPRESSION_PENALTY=0.2

export RUN_TAG="8sess_qwen_judge"

bash "$SCRIPT_DIR/vllm_client_standalone.sh"
