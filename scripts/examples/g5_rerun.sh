#!/bin/bash
export DATASET_DIR=$(pwd)/data/locomo/processed
export PROJECT_NAME="rema-ablation"
export LOG_DIR=$(pwd)/logs/G5_RERUN
mkdir -p $LOG_DIR

# Activate environment
source <workspace>/miniconda3/etc/profile.d/conda.sh
conda activate rema

# Launch training with separated trainer
# Node selection is handled by the user's srun request or I'll use -w hkn1970
PYTHONUNBUFFERED=1 python -m verl.rema_separated_trainer.main_ppo \
    +trainer.insert_penalty=0 \
    +trainer.update_bonus=0 \
    +trainer.delete_bonus=0 \
    algorithm.switch_agent.enable=True \
    algorithm.switch_agent.model_paths=[checkpoints/rema-curriculum-v1/8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed,checkpoints/rema-curriculum-v1/8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed] \
    algorithm.switch_agent.level=step \
    algorithm.switch_agent.freq=2 \
    trainer.experiment_name=8sess_separated_params_switch2_turns4_rerun_$(date +%Y%m%d_%H%M) \
    trainer.project_name=$PROJECT_NAME \
    trainer.total_training_steps=10 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    +trainer.max_sessions=8 \
    data.train_files=$DATASET_DIR/train.parquet \
    data.val_files=$DATASET_DIR/val.parquet \
    +data.test_files=$DATASET_DIR/test.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    actor_rollout_ref.model.path=checkpoints/rema-curriculum-v1/8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed \
    reward_model.reward_manager=rema \
    +trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.logger='["console","wandb"]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.inner_sampling_fraction=0.25 \
    +actor_rollout_ref.rollout.inner_n=4 \
    +trainer.rewardtype=persession \
    +trainer.compression_penalty=0.2 \
    trainer.total_epochs=10 | tee $LOG_DIR/launch.log
