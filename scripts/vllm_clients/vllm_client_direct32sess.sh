#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Client / training node script.
# Waits for the vLLM server to be ready, then runs the full curriculum training.
#
# Run on the training node:
#   srun --jobid=<training_job_id> bash "$SCRIPT_DIR/vllm_client_standalone.sh"

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_JUDGE_MODEL=${VLLM_JUDGE_MODEL:-"openai/gpt-oss-120b"}
RENDEZVOUS_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers
# How long to wait for the first server to appear, and how long to wait
# after the first server for any additional servers to register.
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-600}

# ---------------------------------------------------------------------------
# Common environment
# ---------------------------------------------------------------------------
export JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
export DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/data/locomo/processed_${JOB_ID}
export LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/$JOB_ID
export TMPDIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tmp
export RAY_TMPDIR=/scratch/$USER/ray_$JOB_ID
export HYDRA_RUN_DIR=/scratch/$USER/hydra_$JOB_ID
export SCRATCH_DIR=/scratch/$USER/verl_$JOB_ID
export HF_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_home
export HF_DATASETS_CACHE=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/hf_datasets
export TRITON_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/trition
export TRITON_DUMP_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/trition_dump
export EMBEDDING_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/embedding_cache
export MEMORY_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_$JOB_ID/train
export MEMORY_CACHE_DIR_VAL=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_$JOB_ID/validation
export MEMORY_CACHE_DIR_TEST=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/memory/memory_cache_$JOB_ID/test
export OPENAI_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/openai_cache
export TEACHER_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/teacher_cache
mkdir -p $LOG_DIR $TMPDIR $RAY_TMPDIR $HYDRA_RUN_DIR $SCRATCH_DIR \
         $HF_HOME $HF_DATASETS_CACHE $TRITON_HOME $TRITON_DUMP_DIR \
         $EMBEDDING_CACHE_DIR $MEMORY_CACHE_DIR $MEMORY_CACHE_DIR_VAL \
         $OPENAI_CACHE_DIR $TEACHER_CACHE_DIR

export HYDRA_FULL_ERROR=1
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES

source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/etc/profile.d/conda.sh
conda activate rema

cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

# ---------------------------------------------------------------------------
# Auto-detect servers: wait for the first server_*.txt, then settle
# ---------------------------------------------------------------------------
WAIT_INTERVAL=5
echo "[client] Waiting for at least one server to register in $RENDEZVOUS_DIR ..."
elapsed=0
while true; do
    found=$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null | wc -l)
    [ "$found" -gt 0 ] && break
    if [ "$elapsed" -ge "$SERVER_WAIT_TIMEOUT" ]; then
        echo "[client] ERROR: No servers registered after ${SERVER_WAIT_TIMEOUT}s. Aborting."
        exit 1
    fi
    echo "[client] ... no servers yet (${elapsed}s elapsed)"
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
done

SERVER_FILES=$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null)
NUM_SERVERS=$(echo "$SERVER_FILES" | wc -l)
echo "[client] Auto-detected $NUM_SERVERS server(s)."

# ---------------------------------------------------------------------------
# Health-check all detected servers, build JUDGE_BASE_URLS
# ---------------------------------------------------------------------------
BASE_URLS=""
idx=0
for f in $SERVER_FILES; do
    HOST_PORT=$(cat "$f")
    URL="http://${HOST_PORT}/v1"
    echo "[client] Waiting for server $idx ($f) at $URL ..."
    elapsed=0
    while true; do
        if curl -sf "${URL}/models" > /dev/null 2>&1; then
            echo "[client] Server $idx ready after ${elapsed}s."
            break
        fi
        if [ "$elapsed" -ge "$SERVER_WAIT_TIMEOUT" ]; then
            echo "[client] WARNING: Server $idx did not become ready within ${SERVER_WAIT_TIMEOUT}s. Skipping."
            URL=""
            break
        fi
        sleep "$WAIT_INTERVAL"
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
    [ -n "$URL" ] && BASE_URLS="${BASE_URLS:+${BASE_URLS},}${URL}"
    idx=$((idx + 1))
done

if [ -z "$BASE_URLS" ]; then
    echo "[client] ERROR: No servers became healthy. Aborting."
    exit 1
fi

echo "[client] Using $NUM_SERVERS server(s). URLs: $BASE_URLS"

# ---------------------------------------------------------------------------
# Point judge_llm.py at the local vLLM servers (round-robin)
# ---------------------------------------------------------------------------
export JUDGE_PROVIDER=openai
export JUDGE_BASE_URLS="$BASE_URLS"
export JUDGE_API_KEY="EMPTY"        # vLLM accepts any non-empty key
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"
export OPENAI_JUDGE_MODEL="$VLLM_JUDGE_MODEL"

# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------
export PROJECT_NAME="rema-curriculum-v1"
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CURRENT_MODEL_PATH=$BASE_MODEL

MAX_NUM_TURNS=4
# We have total 10 convs, now we use split 118 -> check data_preprocess.py if you want to change, since 
# we fixed the convs used in each split. (for now can do 118, or 217 without change indata_preprocess.py, other than this we need to change the conv splits in data_preprocess.py)
NUM_TRAIN_CONVS=1
NUM_VAL_CONVS=1
NUM_TEST_CONVS=8
algorithm=grpo
num_rollouts=16
max_prompt_length=24576
max_response_length=2048
use_kl_loss=True
kl_loss_coef=0.001
use_kl_in_reward=False
kl_coef=0.0005
LAM_TOKEN_LEVEL=1.0
GAMMA_TURN_LEVEL=1.0
TEST_SAVE_FREQ=5
PPO_EPOCHS=2
REWARD_TYPE=persession
COMPRESSION_PENALTY=0.2
CLIP_RATIO=0.2

FAST_EXPERIMENT=${FAST_EXPERIMENT:-1}
if [ "$FAST_EXPERIMENT" = "1" ]; then
    export REMA_REWARD_MAX_QA_TRAIN_INNER=${REMA_REWARD_MAX_QA_TRAIN_INNER:-16}
    export REMA_REWARD_MAX_QA_TRAIN_TERMINAL=${REMA_REWARD_MAX_QA_TRAIN_TERMINAL:-64}
    export REMA_REWARD_MAX_QA_EVAL=${REMA_REWARD_MAX_QA_EVAL:-0}
    export REMA_REWARD_QA_SAMPLE_STRATEGY=${REMA_REWARD_QA_SAMPLE_STRATEGY:-random}
    export REMA_REWARD_SAME_QAS_ACROSS_ROLLOUTS=${REMA_REWARD_SAME_QAS_ACROSS_ROLLOUTS:-1}
    export REMA_REWARD_QA_SAMPLE_SEED=${REMA_REWARD_QA_SAMPLE_SEED:-12345}
    export REMA_REWARD_MAX_OUTER_WORKERS=${REMA_REWARD_MAX_OUTER_WORKERS:-8}
    export REMA_REWARD_MAX_INNER_WORKERS=${REMA_REWARD_MAX_INNER_WORKERS:-4}
    export REMA_REWARD_TIMEOUT_S=${REMA_REWARD_TIMEOUT_S:-600}
fi

if [ "$use_kl_loss" = "True" ] && [ "$use_kl_in_reward" = "True" ]; then
    echo "Error: Both use_kl_loss and use_kl_in_reward cannot be True"
    exit 1
fi

export DATA_SEED=$(echo "$JOB_ID" | cksum | awk '{print $1}')
echo "Generated DATA_SEED: $DATA_SEED"

STAGES=(32)
EPOCHS_PER_STAGE=(5)
QA_TOP_K_PER_STAGE=(50)  # per speaker: 8-sess=30, 16-sess=50, 32-sess=50
INNER_GPRO_FRAC=0.5

for i in "${!STAGES[@]}"; do
    STAGE_SESSIONS="${STAGES[$i]}"
    STAGE_EPOCHS="${EPOCHS_PER_STAGE[$i]:-${EPOCHS_PER_STAGE[-1]}}"
    export REMA_REWARD_QA_TOP_K_PER_SPEAKER="${QA_TOP_K_PER_STAGE[$i]:-${QA_TOP_K_PER_STAGE[-1]}}"
    echo "STARTING STAGE: $STAGE_SESSIONS Sessions, $STAGE_EPOCHS Epochs, QA_TOP_K_PER_SPEAKER=$REMA_REWARD_QA_TOP_K_PER_SPEAKER"

    export STAGE_EXP_NAME="curr_${STAGE_SESSIONS}sess_${JOB_ID}_${MAX_NUM_TURNS}turns_${PPO_EPOCHS}ppo_Kl${kl_loss_coef}_${REWARD_TYPE}_${COMPRESSION_PENALTY}addcomp_turn_${algorithm}_${NUM_TRAIN_CONVS}convs${num_rollouts}r_innergrpo${INNER_GPRO_FRAC}sampleQA_pen0oss120b"
    export HYDRA_RUN_DIR=$TMPDIR/hydra_${STAGE_EXP_NAME}
    mkdir -p $HYDRA_RUN_DIR

    python data/locomo/data_preprocess.py --max_sessions $STAGE_SESSIONS --train_convs $NUM_TRAIN_CONVS --val_convs $NUM_VAL_CONVS --test_convs $NUM_TEST_CONVS --seed $DATA_SEED --output_dir $DATASET_DIR

    STAGE_LR="2e-6"

    PYTHONUNBUFFERED=1 python -m verl.rema_trainer.main_ppo \
        +trainer.insert_penalty=0 \
        +trainer.update_bonus=0 \
        +trainer.delete_bonus=0 \
        +actor_rollout_ref.rollout.inner_sampling_fraction=$INNER_GPRO_FRAC \
        +actor_rollout_ref.rollout.inner_n=4 \
        +algorithm.use_bilevel_gae=False \
        actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
        +trainer.rewardtype=$REWARD_TYPE \
        +trainer.compression_penalty=$COMPRESSION_PENALTY \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$STAGE_EXP_NAME \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        +trainer.max_sessions=$STAGE_SESSIONS \
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
        actor_rollout_ref.model.path=$CURRENT_MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        algorithm.lam_token_level=$LAM_TOKEN_LEVEL \
        algorithm.gamma_turn_level=$GAMMA_TURN_LEVEL \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=$num_rollouts \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.rollout.stop_when_truncated=True \
        actor_rollout_ref.actor.optim.lr=$STAGE_LR \
        +trainer.val_before_train=True \
        +trainer.test_before_train=False \
        +trainer.test_after_train=False \
        +trainer.val_only=False \
        +trainer.save_val_generations=True \
        +trainer.save_train_generations=True \
        +trainer.test_only=False \
        trainer.test_freq=$TEST_SAVE_FREQ \
        trainer.save_freq=$TEST_SAVE_FREQ \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=$STAGE_EPOCHS \
        trainer.total_training_steps=$STAGE_EPOCHS \
        algorithm.adv_estimator=$algorithm \
        reward_model.reward_manager=rema \
        reward_model.mask_unfinished_reward=True \
        algorithm.filter_groups.enable=True \
        trainer.logger='["console","wandb"]' \
        hydra.run.dir=$HYDRA_RUN_DIR >> $LOG_DIR/${STAGE_EXP_NAME}.log 2>&1

    CHECKPOINT_ROOT="./checkpoints/$PROJECT_NAME/$STAGE_EXP_NAME"
    ITER_FILE="$CHECKPOINT_ROOT/best_checkpoint_info.txt"

    if [ -f "$ITER_FILE" ]; then
        LATEST_STEP=$(cat "$ITER_FILE")
        SHARDED_PATH="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/actor"
        CONSOLIDATED_PATH="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/hf_fixed"

        python convert_fsdp_to_hf.py \
            --fsdp_checkpoint_path $SHARDED_PATH \
            --huggingface_model_path $SHARDED_PATH/huggingface \
            --output_path $CONSOLIDATED_PATH \
            --world_size 4

        CURRENT_MODEL_PATH=$CONSOLIDATED_PATH
        echo "Stage Complete. Next model: $CURRENT_MODEL_PATH"
    else
        echo "Error: Could not find checkpoint for stage $STAGE_SESSIONS"
        exit 1
    fi
done

echo "[client] Training complete. Skipping final test (val-only mode)."
exit 0

# ---------------------------------------------------------------------------
# Final testing phase (best 32-session model) — DISABLED (use val metrics)
# ---------------------------------------------------------------------------
echo "STARTING FINAL TESTING ON 32 SESSIONS"

python data/locomo/data_preprocess.py --max_sessions 32 --train_convs $NUM_TRAIN_CONVS --val_convs $NUM_VAL_CONVS --test_convs $NUM_TEST_CONVS --seed $DATA_SEED

export TEST_EXP_NAME="TEST_${STAGE_EXP_NAME}"
FINAL_TEST_SESSIONS=32
export HYDRA_RUN_DIR=$TMPDIR/hydra_${TEST_EXP_NAME}
mkdir -p $HYDRA_RUN_DIR

PYTHONUNBUFFERED=1 python -m verl.rema_trainer.main_ppo \
        +actor_rollout_ref.rollout.inner_sampling_fraction=$INNER_GPRO_FRAC \
        +actor_rollout_ref.rollout.inner_n=4 \
        +algorithm.use_bilevel_gae=False \
        actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
        +trainer.rewardtype=$REWARD_TYPE \
        +trainer.compression_penalty=$COMPRESSION_PENALTY \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$TEST_EXP_NAME \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        +trainer.max_sessions=$FINAL_TEST_SESSIONS \
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
        actor_rollout_ref.model.path=$CURRENT_MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        algorithm.lam_token_level=$LAM_TOKEN_LEVEL \
        algorithm.gamma_turn_level=$GAMMA_TURN_LEVEL \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=$num_rollouts \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.rollout.stop_when_truncated=True \
        actor_rollout_ref.actor.optim.lr=$STAGE_LR \
        +trainer.val_before_train=True \
        +trainer.val_only=False \
        +trainer.save_val_generations=True \
        +trainer.save_train_generations=True \
        +trainer.test_only=True \
        trainer.test_freq=$TEST_SAVE_FREQ \
        trainer.save_freq=$TEST_SAVE_FREQ \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=$STAGE_EPOCHS \
        trainer.total_training_steps=$STAGE_EPOCHS \
        algorithm.adv_estimator=$algorithm \
        reward_model.reward_manager=rema \
        reward_model.mask_unfinished_reward=True \
        algorithm.filter_groups.enable=True \
        trainer.logger='["console","wandb"]' \
        hydra.run.dir=$HYDRA_RUN_DIR 2>&1 | tee $LOG_DIR/${TEST_EXP_NAME}.log

echo "[client] Training and final testing complete."
