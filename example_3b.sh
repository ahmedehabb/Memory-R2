#!/bin/bash
export DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/projects/ReMA-public/data/locomo/processed
export JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
export LOG_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/projects/ReMA-public/logs/$JOB_ID
export TMPDIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/tmp
export RAY_TMPDIR=/scratch/$USER/ray_$JOB_ID
export HYDRA_RUN_DIR=/scratch/$USER/hydra_$JOB_ID
export SCRATCH_DIR=/scratch/$USER/verl_$JOB_ID
export HF_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/hf_home
export HF_DATASETS_CACHE=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/hf_datasets
export TRITON_HOME=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/trition
export TRITON_DUMP_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/trition_dump
export EMBEDDING_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/embedding_cache
export MEMORY_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/memory/memory_cache_$JOB_ID/train
export MEMORY_CACHE_DIR_VAL=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/memory/memory_cache_$JOB_ID/validation
export OPENAI_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/openai_cache
export TEACHER_CACHE_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/teacher_cache
mkdir -p $LOG_DIR $TMPDIR $RAY_TMPDIR $HYDRA_RUN_DIR $SCRATCH_DIR $HF_HOME $HF_DATASETS_CACHE $TRITON_HOME $TRITON_DUMP_DIR $EMBEDDING_CACHE_DIR $MEMORY_CACHE_DIR $MEMORY_CACHE_DIR_VAL $OPENAI_CACHE_DIR $TEACHER_CACHE_DIR

export HYDRA_FULL_ERROR=1
export HF_TOKEN="***HF_TOKEN_REDACTED***"
export OPENAI_API_KEY="***OPENAI_KEY_REDACTED***"
export GEMINI_API_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY via env or sourced .env file}"
# new: ${GEMINI_API_KEY:?Set GEMINI_API_KEY via env or sourced .env file} ---- old: ${GEMINI_API_KEY:?Set GEMINI_API_KEY via env or sourced .env file}
export WANDB_API_KEY="***WANDB_KEY_REDACTED***"
unset ROCR_VISIBLE_DEVICES
cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace/projects/ReMA-public

echo "🔹 Activating Python environment..."
source /hkfs/work/workspace/scratch/tum_eyi5958-myspace/miniconda3/etc/profile.d/conda.sh
conda activate rema

which python  # should be 3.11.7
which ray     # should be the one inside flashenv

echo "🔹 Launching PPO training..."

MAX_NUM_TURNS=6
CHUNK_SIZE=0  # Number of dialogue turns per chunk
NUM_TRAIN_CONVS=4
PROJECT_NAME=rema-test
DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace/projects/ReMA-public/data/locomo/processed
algorithm=grpo
num_rollouts=8
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
SPLIT=415
EXPERIMENT_NAME=${MODEL_PATH}_${SPLIT}_${MAX_NUM_TURNS}_separate

# Define lengths per turn (16384 hoping it will fit into 32768 context length with 4 turns) 
# But did it this way since some turns may be shorter/longer than others to avoid completion_token_exceeded
prompt_length_per_turn=16384
response_length_per_turn=16384

# Set max lengths according to the model's context length limit
max_prompt_length=16384
max_response_length=16384

# *2 since per turn we have reasoning + executor !! no since 2 roles are not concatenated together
# max_prompt_length=$((prompt_length_per_turn * MAX_NUM_TURNS))
# max_response_length=$((response_length_per_turn * MAX_NUM_TURNS))

# if (( max_prompt_length + max_response_length > 32768 )); then
#     echo "⚠️  Warning: The total length of prompt and response exceeds 32768 tokens, which may lead to out-of-memory issues."
#     exit
# fi

# Set this to false, since we dont generate [finished] tokens in multi-turn setting, so no need to mask unfinished rewards
# reward_model.mask_unfinished_reward=False \

PYTHONUNBUFFERED=1 python -m verl.rema_trainer.main_ppo \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.train_files=$DATASET_DIR/train.parquet \
    data.val_files=$DATASET_DIR/val.parquet \
    data.val_batch_size=1 \
    data.train_batch_size=$NUM_TRAIN_CONVS \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.rollout.prompt_length=$prompt_length_per_turn \
    actor_rollout_ref.rollout.response_length=$response_length_per_turn \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.clip_mode=turn \
    actor_rollout_ref.actor.agg_mode=trajectory \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
    actor_rollout_ref.rollout.n=$num_rollouts \
    actor_rollout_ref.rollout.stop_when_truncated=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +trainer.val_before_train=True \
    +trainer.val_only=False \
    +trainer.save_val_generations=True \
    +trainer.save_train_generations=True \
    +trainer.test_only=False \
    trainer.test_freq=15 \
    trainer.save_freq=15 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    algorithm.adv_estimator=$algorithm \
    reward_model.reward_manager=rema \
    reward_model.mask_unfinished_reward=False \
    algorithm.filter_groups.enable=False \
    trainer.logger='["console","wandb"]' \
    hydra.run.dir=$HYDRA_RUN_DIR 2>&1 | tee $LOG_DIR/ppo.log