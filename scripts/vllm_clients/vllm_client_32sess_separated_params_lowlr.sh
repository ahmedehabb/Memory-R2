#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Separated-parameters ablation (G5): two independent models (one per role),
# alternately frozen every 10 steps.
#
# Architecture: Agent0 (meta_thinking) and Agent1 (reasoning) each get 2 GPUs
# on the same node (n_gpus_per_node=2 × 2 pools = 4 GPUs total).
# Both start from the same base model (Qwen2.5-7B-Instruct) — cleanest comparison.
#
# Baseline: inner_n8_rerun (shared params, turns=4, val=0.488, mfail=0.050)
# Claim test: co-learning (shared params) > separated training
#
# Run on the training node:
#   srun --jobid=<training_job_id> bash "$SCRIPT_DIR/vllm_client_8sess_separated_params.sh"
#
# NOTE: requires a vLLM judge server running. Will auto-detect from vllm_servers/*.txt

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_JUDGE_MODEL=${VLLM_JUDGE_MODEL:-"openai/gpt-oss-120b"}
RENDEZVOUS_DIR=${RENDEZVOUS_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-600}

# ---------------------------------------------------------------------------
# Common environment
# ---------------------------------------------------------------------------
export JOB_ID=${JOB_ID:-sep_params}
export DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/data/locomo/processed_${JOB_ID}
export LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/$JOB_ID
export TMPDIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/tmp
export RAY_TMPDIR=${RAY_TMPDIR:-/scratch/$USER/ray}
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

export PATH=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/envs/rema/bin:$PATH

cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

# ---------------------------------------------------------------------------
# Auto-detect servers
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
            echo "[client] WARNING: Server $idx did not become ready. Skipping."
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

export JUDGE_PROVIDER=openai
export JUDGE_BASE_URLS="$BASE_URLS"
export JUDGE_API_KEY="EMPTY"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"
export OPENAI_JUDGE_MODEL="$VLLM_JUDGE_MODEL"

# ---------------------------------------------------------------------------
# Separated-params ablation (8-sess, matched to inner_n8_rerun)
# Key difference from inner_n8_rerun:
#   - rema_separated_trainer (two independent model pools)
#   - n_gpus_per_node=2 (2 GPUs per agent, 4 GPUs total on one node)
#   - switch_agent alternates frozen every 10 steps
#   - Both agents start from same base model (cleanest ablation)
# ---------------------------------------------------------------------------
export PROJECT_NAME=${PROJECT_NAME:-"rema-curriculum-v1"}
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

MAX_NUM_TURNS=4
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
INNER_GPRO_FRAC=0.5
# Switch frequency can be overridden from env (paper rerun uses 2).
SWITCH_FREQ=${SWITCH_FREQ:-1}
START_AGENT=${START_AGENT:-meta_thinking}

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

export DATA_SEED=$(echo "$JOB_ID" | cksum | awk '{print $1}')
echo "Generated DATA_SEED: $DATA_SEED"

STAGE_SESSIONS=32
TOTAL_STEPS=${TOTAL_STEPS:-5}
STAGE_EPOCHS=$TOTAL_STEPS
export REMA_REWARD_QA_TOP_K_PER_SPEAKER=30

echo "STARTING separated-params ablation: $STAGE_SESSIONS Sessions, $STAGE_EPOCHS Epochs"

export STAGE_EXP_NAME="32sess_separated_n5_LOWLR_switch${SWITCH_FREQ}_start${START_AGENT}_turns${MAX_NUM_TURNS}_${PPO_EPOCHS}ppo_Kl${kl_loss_coef}_${REWARD_TYPE}_${COMPRESSION_PENALTY}addcomp_turn_${algorithm}_${NUM_TRAIN_CONVS}convs${num_rollouts}r_innergrpo${INNER_GPRO_FRAC}"
export HYDRA_RUN_DIR=$TMPDIR/hydra_${STAGE_EXP_NAME}
mkdir -p $HYDRA_RUN_DIR

python data/locomo/data_preprocess.py --max_sessions $STAGE_SESSIONS --train_convs $NUM_TRAIN_CONVS --val_convs $NUM_VAL_CONVS --test_convs $NUM_TEST_CONVS --seed $DATA_SEED --output_dir $DATASET_DIR

STAGE_LR="1e-6"

PYTHONUNBUFFERED=1 python -m verl.rema_separated_trainer.main_ppo \
    +trainer.insert_penalty=0 \
    +trainer.update_bonus=0 \
    +trainer.delete_bonus=0 \
    +actor_rollout_ref.rollout.inner_sampling_fraction=$INNER_GPRO_FRAC \
    +actor_rollout_ref.rollout.inner_n=8 \
    +algorithm.use_bilevel_gae=False \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    +trainer.rewardtype=$REWARD_TYPE \
    +trainer.compression_penalty=$COMPRESSION_PENALTY \
    algorithm.switch_agent.enable=True \
    algorithm.switch_agent.level=step \
    algorithm.switch_agent.freq=$SWITCH_FREQ \
    algorithm.switch_agent.model_paths=[/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/8sess_separated_params_switch10_turns4_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5/global_step_20/meta_thinking/actor/hf_fixed,/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/checkpoints/rema-curriculum-v1/8sess_separated_params_switch10_turns4_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5/global_step_20/reasoning/actor/hf_fixed] \
    algorithm.switch_agent.start_agent=$START_AGENT \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$STAGE_EXP_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
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
    actor_rollout_ref.model.path=$BASE_MODEL \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_mode=turn \
    actor_rollout_ref.actor.agg_mode=turn \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length+max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length+max_response_length)) \
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

echo "[client] Separated-params ablation training complete."
echo "[client] Compare val/acc and mfail against shared-params baseline: inner_n8_rerun (val=0.488); 16-sess separated baseline yn1sucq6 (val=0.484)"
echo "[client] Log: $LOG_DIR/${STAGE_EXP_NAME}.log"
