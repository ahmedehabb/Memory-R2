#!/bin/bash
export DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/data/locomo/processed
export JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
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
mkdir -p $LOG_DIR $TMPDIR $RAY_TMPDIR $HYDRA_RUN_DIR $SCRATCH_DIR $HF_HOME $HF_DATASETS_CACHE $TRITON_HOME $TRITON_DUMP_DIR $EMBEDDING_CACHE_DIR $MEMORY_CACHE_DIR $MEMORY_CACHE_DIR_VAL $OPENAI_CACHE_DIR $TEACHER_CACHE_DIR

# Generate a stable data seed from JOB_ID for consistency within a run
export DATA_SEED=$(echo "$JOB_ID" | cksum | awk '{print $1}')
echo "🎲 Generated DATA_SEED: $DATA_SEED"

export HYDRA_FULL_ERROR=1
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"
export GEMINI_API_KEY="${GEMINI_API_KEY:?Set GEMINI_API_KEY via env or sourced .env file}"
# export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY via env or sourced .env file}"
# [REDACTED] :: sikuans one
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY via env or sourced .env file}" #paid one
export JUDGE_PROVIDER=together
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES
cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

echo "🔹 Activating Python environment..."
source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/etc/profile.d/conda.sh
conda activate rema

export PROJECT_NAME="rema-curriculum-v1"
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CURRENT_MODEL_PATH=$BASE_MODEL

# Fixed hyperparams from your snippet
MAX_NUM_TURNS=4
NUM_TRAIN_CONVS=1
NUM_VAL_CONVS=1
NUM_TEST_CONVS=8
algorithm=grpo
num_rollouts=8
max_prompt_length=16384
max_response_length=2048
use_kl_loss=True
kl_loss_coef=0.01
use_kl_in_reward=False
kl_coef=0.0005
LAM_TOKEN_LEVEL=1.0
GAMMA_TURN_LEVEL=1.0
REPLAY_MIX_RATIO=0
TEST_SAVE_FREQ=5 # Adjusted slightly for curriculum logic
NUM_EPOCHS=50
PPO_EPOCHS=2
REWARD_TYPE=persession # cumulative or persession or global
COMPRESSION_PENALTY=0.2
CLIP_RATIO=0.2

# Fast reward mode for quick iteration.
# Set FAST_EXPERIMENT=0 to evaluate all QA pairs with default parallelism.
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

# assert if both kl are true
if [ "$use_kl_loss" = "True" ] && [ "$use_kl_in_reward" = "True" ]; then
    echo "Error: Both use_kl_loss and use_kl_in_reward cannot be True"
    exit 1
fi


# --- 2. Curriculum Definition ---
# The progression: 2 -> 4 -> 8 -> 16 -> 32
STAGES=(32)
INNER_GPRO_FRAC=0.25 # Fraction of rollouts in each stage that use inner GRPO sampling

for STAGE_SESSIONS in "${STAGES[@]}"; do
    echo "🚀 STARTING STAGE: $STAGE_SESSIONS Sessions"
    
    export STAGE_EXP_NAME="curr_${STAGE_SESSIONS}sess_${JOB_ID}_${MAX_NUM_TURNS}turns_${PPO_EPOCHS}ppo_Kl${kl_loss_coef}_${REWARD_TYPE}_${COMPRESSION_PENALTY}addcomp_turn_${algorithm}_${NUM_TRAIN_CONVS}convs${num_rollouts}r_innergrpo${INNER_GPRO_FRAC}sampleQA_vanillaoss120b"
    export HYDRA_RUN_DIR=$TMPDIR/hydra_${STAGE_EXP_NAME}
    mkdir -p $HYDRA_RUN_DIR

    # A. Preprocess for this stage
    python data/locomo/data_preprocess.py --max_sessions $STAGE_SESSIONS --train_convs $NUM_TRAIN_CONVS --val_convs $NUM_VAL_CONVS --test_convs $NUM_TEST_CONVS --seed $DATA_SEED

    STAGE_LR="2e-6"

    # actor_rollout_ref.actor.clip_ratio=0.02 \ as adviced by rema repo to be 1-2 orders of magnitude less than token level clipping.

    # B. The Main PPO Launch (Using your EXACT flags)
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
        +trainer.replay_mix_ratio=$REPLAY_MIX_RATIO \
        +trainer.replay_strategy=uniform \
        +trainer.replay_buffer_size=0 \
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
        actor_rollout_ref.actor.entropy_coeff=0.001  \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=$num_rollouts \
        actor_rollout_ref.rollout.val_kwargs.n=2 \
        actor_rollout_ref.rollout.stop_when_truncated=True \
        actor_rollout_ref.actor.optim.lr=$STAGE_LR \
        +trainer.val_before_train=True \
        +trainer.val_only=False \
        +trainer.save_val_generations=True \
        +trainer.save_train_generations=True \
        +trainer.test_only=False \
        trainer.test_freq=$TEST_SAVE_FREQ \
        trainer.save_freq=$TEST_SAVE_FREQ \
        trainer.remove_previous_ckpt_in_save=False \
        trainer.total_epochs=$NUM_EPOCHS \
        trainer.total_training_steps=$NUM_EPOCHS \
        algorithm.adv_estimator=$algorithm \
        reward_model.reward_manager=rema \
        reward_model.mask_unfinished_reward=True \
        algorithm.filter_groups.enable=True \
        trainer.logger='["console","wandb"]' \
        hydra.run.dir=$HYDRA_RUN_DIR 2>&1 | tee $LOG_DIR/${STAGE_EXP_NAME}.log

    # D. Path Discovery for Next Iteration
    # Structure: ./checkpoints/project/experiment/global_step_X
    CHECKPOINT_ROOT="./checkpoints/$PROJECT_NAME/$STAGE_EXP_NAME"
    # Now we are using the best checkpoint on validation from last stage.
    ITER_FILE="$CHECKPOINT_ROOT/best_checkpoint_info.txt"

    if [ -f "$ITER_FILE" ]; then
        LATEST_STEP=$(cat "$ITER_FILE")
        SHARDED_PATH="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/actor"
        CONSOLIDATED_PATH="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/hf_fixed"

        # 🔹 CALL CONVERSION HERE
        python convert_fsdp_to_hf.py \
            --fsdp_checkpoint_path $SHARDED_PATH \
            --huggingface_model_path $SHARDED_PATH/huggingface \
            --output_path $CONSOLIDATED_PATH \
            --world_size 4  # Pass the world size dynamically

        # 🔹 Update path for next stage to the CONSOLIDATED one
        CURRENT_MODEL_PATH=$CONSOLIDATED_PATH
        echo "✅ Stage Complete. Next model: $CURRENT_MODEL_PATH"
    else
        echo "❌ Error: Could not find checkpoint for stage $STAGE_SESSIONS"
        exit 1
    fi
done


# ==========================================
# 🏆 FINAL TESTING PHASE (BEST 32-SESSION MODEL)
# ==========================================
echo "🚀 STARTING FINAL TESTING ON 32 SESSIONS"

# # Ensure the data is strictly preprocessed for 32 sessions (in case it got reset)
python data/locomo/data_preprocess.py --max_sessions 32 --train_convs $NUM_TRAIN_CONVS --val_convs $NUM_VAL_CONVS --test_convs $NUM_TEST_CONVS --seed $DATA_SEED

export TEST_EXP_NAME="TEST_${STAGE_EXP_NAME}"
FINAL_TEST_SESSIONS=32
export HYDRA_RUN_DIR=$TMPDIR/hydra_${TEST_EXP_NAME}
mkdir -p $HYDRA_RUN_DIR

# Run the identical configuration, but with test_only=True and no training epochs
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
        +trainer.replay_mix_ratio=$REPLAY_MIX_RATIO \
        +trainer.replay_strategy=uniform \
        +trainer.replay_buffer_size=0 \
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
        actor_rollout_ref.actor.entropy_coeff=0.001  \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length+max_response_length) * 2)) \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=$num_rollouts \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
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
        trainer.total_epochs=$NUM_EPOCHS \
        trainer.total_training_steps=$NUM_EPOCHS \
        algorithm.adv_estimator=$algorithm \
        reward_model.reward_manager=rema \
        reward_model.mask_unfinished_reward=True \
        algorithm.filter_groups.enable=True \
        trainer.logger='["console","wandb"]' \
        hydra.run.dir=$HYDRA_RUN_DIR 2>&1 | tee $LOG_DIR/${TEST_EXP_NAME}.log

# echo "🎉 ALL STAGES AND FINAL TESTING COMPLETE."
