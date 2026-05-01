#!/bin/bash
# Test set evaluation — uses Qwen2.5-7B-Instruct as judge (NOT gpt-oss-120b).
# Connects ONLY to the isolated vllm_servers_qwen/ rendezvous dir.
# Run vllm_server_qwen.sh first on a separate job allocation.
#
# Usage:
#   export MODEL_PATH_OVERRIDE="<hf_fixed_checkpoint_path>"   # omit = Qwen base
#   export RUN_TAG="qwen_judge_champion_v2"
#   srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_test_eval_qwen.sh > logs/<job_id>/test_qwen_launch.log 2>&1 &
#
# Preset shortcuts:
#   32sess_champion_v2:
#     export MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
#     export RUN_TAG="qwen_judge_champion_v2"
#
#   32sess_inner0:
#     export MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_inner0__20260402_022134_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_5/hf_fixed"
#     export RUN_TAG="qwen_judge_inner0"
#
#   Untrained baseline:
#     (omit MODEL_PATH_OVERRIDE)
#     export RUN_TAG="qwen_judge_baseline"

export SKIP_NODE_CHECK=1

# ---------------------------------------------------------------------------
# Qwen judge — isolated rendezvous dir (never mix with vllm_servers/)
# ---------------------------------------------------------------------------
VLLM_PORT=${VLLM_PORT:-8100}
RENDEZVOUS_DIR=${JUDGE_RENDEZVOUS_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers_qwen}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-600}

export JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
export RUN_TAG=${RUN_TAG:-qwen_judge_eval_${JOB_ID}}
export RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_TAG_SAFE=$(echo "${RUN_TAG}" | tr -cs '[:alnum:]_-' '_')
export LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/$JOB_ID
export TMPDIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tmp
export RAY_TMPDIR=/scratch/$USER/ray_$JOB_ID
export HYDRA_RUN_DIR=/scratch/$USER/hydra_${RUN_TAG_SAFE}_$JOB_ID
export SCRATCH_DIR=/scratch/$USER/verl_$JOB_ID
export HF_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home
export HF_DATASETS_CACHE=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_datasets
export TRITON_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/trition
export TRITON_DUMP_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/trition_dump
export EMBEDDING_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/embedding_cache
export MEMORY_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_${JOB_ID}_${RUN_TAG_SAFE}/train
export MEMORY_CACHE_DIR_VAL=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_${JOB_ID}_${RUN_TAG_SAFE}/validation
export MEMORY_CACHE_DIR_TEST=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_${JOB_ID}_${RUN_TAG_SAFE}/test
export OPENAI_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/openai_cache
export TEACHER_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/teacher_cache

mkdir -p $LOG_DIR $TMPDIR $RAY_TMPDIR $HYDRA_RUN_DIR $SCRATCH_DIR \
         $HF_HOME $HF_DATASETS_CACHE $TRITON_HOME $TRITON_DUMP_DIR \
         $EMBEDDING_CACHE_DIR $MEMORY_CACHE_DIR $MEMORY_CACHE_DIR_VAL $MEMORY_CACHE_DIR_TEST \
         $OPENAI_CACHE_DIR $TEACHER_CACHE_DIR "$RENDEZVOUS_DIR"

export HYDRA_FULL_ERROR=1
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES

source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/etc/profile.d/conda.sh
conda activate rema
export PATH="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/envs/rema/bin:$PATH"

cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

# ---------------------------------------------------------------------------
# Wait for Qwen judge server (vllm_servers_qwen/ only)
# ---------------------------------------------------------------------------
WAIT_INTERVAL=5
echo "[test-eval-qwen] Waiting for Qwen judge server in $RENDEZVOUS_DIR ..."
elapsed=0
while true; do
    found=$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null | wc -l)
    [ "$found" -gt 0 ] && break
    if [ "$elapsed" -ge "$SERVER_WAIT_TIMEOUT" ]; then
        echo "[test-eval-qwen] ERROR: No Qwen judge server registered after ${SERVER_WAIT_TIMEOUT}s."
        echo "[test-eval-qwen] Make sure vllm_server_qwen.sh is running on another job allocation."
        exit 1
    fi
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
done

SERVER_FILES=$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null)
BASE_URLS=""
for f in $SERVER_FILES; do
    HOST_PORT=$(cat "$f")
    URL="http://${HOST_PORT}/v1"
    elapsed=0
    while true; do
        if curl -sf "${URL}/models" > /dev/null 2>&1; then break; fi
        if [ "$elapsed" -ge "$SERVER_WAIT_TIMEOUT" ]; then URL=""; break; fi
        sleep "$WAIT_INTERVAL"; elapsed=$((elapsed + WAIT_INTERVAL))
    done
    [ -n "$URL" ] && BASE_URLS="${BASE_URLS:+${BASE_URLS},}${URL}"
done

if [ -z "$BASE_URLS" ]; then
    echo "[test-eval-qwen] ERROR: No healthy Qwen judge servers."
    exit 1
fi

export JUDGE_PROVIDER=openai
export JUDGE_BASE_URLS="$BASE_URLS"
export JUDGE_RENDEZVOUS_DIR="$RENDEZVOUS_DIR"
export JUDGE_API_KEY="EMPTY"
# Keep your real OpenAI key for embedding retrieval (memory search).
# Judge requests still go to local vLLM via JUDGE_BASE_URLS above.
if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY}" = "EMPTY" ]; then
    FALLBACK_KEY=$(grep -m1 '^export OPENAI_API_KEY=' /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/scripts/vllm_clients/vllm_client_test_eval.sh | cut -d'"' -f2)
    if [ -n "$FALLBACK_KEY" ]; then
        export OPENAI_API_KEY="$FALLBACK_KEY"
        echo "[test-eval-qwen] OPENAI_API_KEY not provided; using fallback from vllm_client_test_eval.sh"
    fi
fi
if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY}" = "EMPTY" ]; then
    echo "[test-eval-qwen] ERROR: OPENAI_API_KEY is required for embedding retrieval."
    echo "[test-eval-qwen] Export a valid key before launch, e.g.:"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# Auto-detect the exact served model ID from the first healthy local vLLM URL.
# This avoids 404s when the server hosts a local checkpoint path as model ID.
FIRST_BASE_URL="${BASE_URLS%%,*}"
OPENAI_JUDGE_MODEL_AUTO=$(python - <<'PY' "$FIRST_BASE_URL"
import json
import sys
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
models_url = f"{base_url}/models"
try:
    with urlopen(models_url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data") or []
    if data and isinstance(data[0], dict) and data[0].get("id"):
        print(data[0]["id"])
except Exception:
    pass
PY
)

if [ -n "${OPENAI_JUDGE_MODEL_AUTO}" ]; then
    export OPENAI_JUDGE_MODEL="$OPENAI_JUDGE_MODEL_AUTO"
else
    export OPENAI_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
fi

# ---------------------------------------------------------------------------
# Eval config (identical to vllm_client_test_eval.sh)
# ---------------------------------------------------------------------------
export PROJECT_NAME="rema-curriculum-v1"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH=${MODEL_PATH_OVERRIDE:-$BASE_MODEL}
echo "[test-eval-qwen] Judge: Qwen2.5-7B-Instruct (local, via $BASE_URLS)"
echo "[test-eval-qwen] Judge model id: $OPENAI_JUDGE_MODEL"
echo "[test-eval-qwen] Model: $MODEL_PATH"
echo "[test-eval-qwen] RUN_TAG: $RUN_TAG"

DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/data/locomo/processed
NUM_TEST_CONVS=7
NUM_VAL_CONVS=1
NUM_TRAIN_CONVS=4

# KV-cache-safe overrides for tighter memory nodes.
max_prompt_length=${MAX_PROMPT_LENGTH_OVERRIDE:-28672}
max_response_length=${MAX_RESPONSE_LENGTH_OVERRIDE:-4096}
MAX_NUM_TURNS=${MAX_NUM_TURNS:-6}
TEST_SESSIONS=32
if [ "${EVAL_SAFE_MODE:-0}" = "1" ]; then
    max_prompt_length=${MAX_PROMPT_LENGTH_OVERRIDE:-24576}
    max_response_length=${MAX_RESPONSE_LENGTH_OVERRIDE:-3072}
fi
rollout_max_batched_tokens=${MAX_NUM_BATCHED_TOKENS_OVERRIDE:-$(((max_prompt_length+max_response_length) * 2))}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU_OVERRIDE:-$(((max_prompt_length+max_response_length) * 2))}

export REMA_REWARD_MAX_QA_EVAL=0
export REMA_REWARD_MAX_QA_TRAIN_INNER=0
export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=0
export REMA_REWARD_QA_SAMPLE_STRATEGY=first
export REMA_REWARD_MAX_OUTER_WORKERS=8
export REMA_REWARD_MAX_INNER_WORKERS=4
export REMA_REWARD_TIMEOUT_S=3600
export REMA_REWARD_QA_TOP_K_PER_SPEAKER=${QA_TOP_K_OVERRIDE:-30}

TEST_EXP_NAME="${RUN_TAG}_${RUN_TS}"
export HYDRA_RUN_DIR=$TMPDIR/hydra_${TEST_EXP_NAME}
mkdir -p $HYDRA_RUN_DIR

echo "[test-eval-qwen] Starting test evaluation on $TEST_SESSIONS sessions, $NUM_TEST_CONVS test convs..."
echo "[test-eval-qwen] Data: $DATASET_DIR"
echo "[test-eval-qwen] prompt=$max_prompt_length response=$max_response_length max_batched_tokens=$rollout_max_batched_tokens"

PYTHONUNBUFFERED=1 python -m verl.rema_trainer.main_ppo \
        +actor_rollout_ref.rollout.inner_sampling_fraction=0.0 \
        +actor_rollout_ref.rollout.inner_n=4 \
        +algorithm.use_bilevel_gae=False \
        actor_rollout_ref.actor.ppo_epochs=2 \
        +trainer.rewardtype=persession \
        +trainer.compression_penalty=0.3 \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$TEST_EXP_NAME \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        +trainer.max_sessions=$TEST_SESSIONS \
        data.train_files=$DATASET_DIR/train.parquet \
        data.val_files=$DATASET_DIR/val.parquet \
        data.val_batch_size=$NUM_VAL_CONVS \
        +data.test_files=$DATASET_DIR/test.parquet \
        +data.test_batch_size=$NUM_TEST_CONVS \
        data.train_batch_size=$NUM_TRAIN_CONVS \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        actor_rollout_ref.rollout.prompt_length=$max_prompt_length \
        actor_rollout_ref.rollout.response_length=$max_response_length \
        data.shuffle=False \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0005 \
        algorithm.lam_token_level=1.0 \
        algorithm.gamma_turn_level=1.0 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_OVERRIDE:-4} \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_OVERRIDE:-4} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_OVERRIDE:-4} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=${EVAL_GPU_MEM_UTIL:-0.85} \
        actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_batched_tokens \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_KWARGS_N:-8} \
        actor_rollout_ref.rollout.stop_when_truncated=True \
        +trainer.val_before_train=True \
        +trainer.val_only=False \
        +trainer.test_before_train=False \
        +trainer.test_after_train=False \
        +trainer.save_val_generations=True \
        +trainer.save_train_generations=False \
        +trainer.test_only=True \
        trainer.resume_mode=disable \
        trainer.resume_from_path=False \
        trainer.test_freq=1 \
        trainer.save_freq=1 \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        algorithm.adv_estimator=grpo \
        reward_model.reward_manager=rema \
        reward_model.mask_unfinished_reward=True \
        algorithm.filter_groups.enable=True \
        trainer.logger='["console","wandb"]' \
        hydra.run.dir=$HYDRA_RUN_DIR >> $LOG_DIR/${TEST_EXP_NAME}.log 2>&1

echo "[test-eval-qwen] Done. Log: $LOG_DIR/${TEST_EXP_NAME}.log"
