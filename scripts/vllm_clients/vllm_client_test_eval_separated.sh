#!/bin/bash
# Test set evaluation script — runs test_only on the 7 held-out conversations.
# Evaluates a trained (or untrained) model without any training updates.
#
# Usage:
#   export MODEL_PATH_OVERRIDE="<hf_fixed_checkpoint_path>"   # omit = Qwen base
#   export RUN_TAG="test_champion_v2"                          # label for logs/wandb
#   srun --jobid=<job_id> --overlap -N1 -n1 bash vllm_client_test_eval.sh > logs/<job_id>/test_launch.log 2>&1 &
#
# Preset shortcuts (copy and paste):
#   32sess_champion_v2:
#     export MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
#     export RUN_TAG="test_champion_v2_inner05"
#
#   32sess_inner0:
#     export MODEL_PATH_OVERRIDE="checkpoints/rema-curriculum-v1/curr_32sess_32sess_inner0__20260402_022134_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_5/hf_fixed"
#     export RUN_TAG="test_inner0_ablation"
#
#   Untrained baseline:
#     (omit MODEL_PATH_OVERRIDE — uses Qwen/Qwen2.5-7B-Instruct base)
#     export RUN_TAG="test_baseline_qwen"

export SKIP_NODE_CHECK=1

# ---------------------------------------------------------------------------
# Source common environment from standalone (env vars, conda, server wait)
# ---------------------------------------------------------------------------
VLLM_PORT=${VLLM_PORT:-8000}
RENDEZVOUS_DIR=${RENDEZVOUS_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/vllm_servers}
SERVER_WAIT_TIMEOUT=${SERVER_WAIT_TIMEOUT:-600}

export JOB_ID=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
export RUN_TAG=${RUN_TAG:-test_eval_${JOB_ID}}
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
         $OPENAI_CACHE_DIR $TEACHER_CACHE_DIR

export HYDRA_FULL_ERROR=1
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN via env or sourced .env file}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY via env or sourced .env file}"
unset ROCR_VISIBLE_DEVICES

source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/etc/profile.d/conda.sh
conda activate rema
export PATH="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/envs/rema/bin:$PATH"

cd /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public

# ---------------------------------------------------------------------------
# Wait for judge server
# ---------------------------------------------------------------------------
WAIT_INTERVAL=5
echo "[test-eval] Waiting for judge server in $RENDEZVOUS_DIR ..."
elapsed=0
while true; do
    found=$(ls "$RENDEZVOUS_DIR"/server_*.txt 2>/dev/null | wc -l)
    [ "$found" -gt 0 ] && break
    if [ "$elapsed" -ge "$SERVER_WAIT_TIMEOUT" ]; then
        echo "[test-eval] ERROR: No judge server registered after ${SERVER_WAIT_TIMEOUT}s."
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
    echo "[test-eval] ERROR: No healthy judge servers."
    exit 1
fi

export JUDGE_PROVIDER=openai
export JUDGE_BASE_URLS="$BASE_URLS"
export JUDGE_RENDEZVOUS_DIR="$RENDEZVOUS_DIR"
export JUDGE_API_KEY="EMPTY"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY via env or sourced .env file}"
export OPENAI_JUDGE_MODEL="openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------
export PROJECT_NAME="rema-curriculum-v1"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH=${MODEL_PATH_OVERRIDE:-$BASE_MODEL}
echo "[test-eval] Model: $MODEL_PATH"
echo "[test-eval] RUN_TAG: $RUN_TAG"

# Use pre-processed data (max_sessions=32, all 7 test convs)
DATASET_DIR=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/data/locomo/processed
NUM_TEST_CONVS=7    # 7 held-out test convs (conv-41,49,50,42,48,30,26); conv-47 reserved for train
NUM_VAL_CONVS=1
NUM_TRAIN_CONVS=4   # must be divisible by n_gpus=4 (trainer validation requirement)

# KV-cache-safe overrides for tighter memory nodes.
# Example usage:
#   export EVAL_SAFE_MODE=1
#   export MAX_PROMPT_LENGTH_OVERRIDE=24576
#   export MAX_RESPONSE_LENGTH_OVERRIDE=3072
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

# For eval: use ALL QAs, no capping
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

# QA-dump support (mirror vllm_client_test_eval.sh): set REMA_DUMP_QA=1 to enable
# post-hoc LLM-judge scoring on per-QA dumps from rema.py reward manager.
export REMA_RUN_NAME=${REMA_RUN_NAME:-${RUN_TAG}_${RUN_TS}}
if [ -n "${REMA_DUMP_QA:-}" ]; then
  export REMA_DUMP_QA
  export REMA_QA_DUMP_DIR=${REMA_QA_DUMP_DIR:-/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/qa_dumps}
  export REMA_QA_DUMP_SPLITS=${REMA_QA_DUMP_SPLITS:-test,val}
  echo "[test-eval-sep] QA-dump enabled: dir=$REMA_QA_DUMP_DIR splits=$REMA_QA_DUMP_SPLITS run=$REMA_RUN_NAME"
fi

echo "[test-eval] Starting test evaluation on $TEST_SESSIONS sessions, $NUM_TEST_CONVS test convs..."
echo "[test-eval] Data: $DATASET_DIR"
echo "[test-eval] prompt=$max_prompt_length response=$max_response_length max_batched_tokens=$rollout_max_batched_tokens"

LOG_FILE="$LOG_DIR/${TEST_EXP_NAME}.log"
RUN_START_EPOCH=$(date +%s)
RUN_START_ISO=$(date -Iseconds)
echo "[test-eval] START_TS=$RUN_START_ISO" | tee -a "$LOG_FILE"

MODEL_PATH_META=${MODEL_PATH_META:-Qwen/Qwen2.5-7B-Instruct}
MODEL_PATH_REASONING=${MODEL_PATH_REASONING:-Qwen/Qwen2.5-7B-Instruct}
START_AGENT=${START_AGENT:-meta_thinking}
SWITCH_FREQ=${SWITCH_FREQ:-100}
echo "[test-eval-sep] meta_thinking=$MODEL_PATH_META"
echo "[test-eval-sep] reasoning=$MODEL_PATH_REASONING"

PYTHONUNBUFFERED=1 python -m verl.rema_separated_trainer.main_ppo \
        +trainer.insert_penalty=0 \
        +trainer.update_bonus=0 \
        +trainer.delete_bonus=0 \
        +actor_rollout_ref.rollout.inner_sampling_fraction=0.0 \
        +actor_rollout_ref.rollout.inner_n=4 \
        +algorithm.use_bilevel_gae=False \
        actor_rollout_ref.actor.ppo_epochs=2 \
        +trainer.rewardtype=persession \
        +trainer.compression_penalty=0.3 \
        algorithm.switch_agent.enable=True \
        algorithm.switch_agent.level=step \
        algorithm.switch_agent.freq=$SWITCH_FREQ \
        "algorithm.switch_agent.model_paths=[$MODEL_PATH_META,$MODEL_PATH_REASONING]" \
        algorithm.switch_agent.start_agent=$START_AGENT \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$TEST_EXP_NAME \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=2 \
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
        actor_rollout_ref.model.path=$MODEL_PATH_META \
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
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.clip_mode=turn \
        actor_rollout_ref.actor.agg_mode=turn \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_batched_tokens \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
        actor_rollout_ref.rollout.max_num_turns=$MAX_NUM_TURNS \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_KWARGS_N:-8} \
        actor_rollout_ref.rollout.stop_when_truncated=True \
        +trainer.val_before_train=False \
        +trainer.val_only=False \
        +trainer.test_before_train=False \
        +trainer.test_after_train=False \
        +trainer.save_val_generations=True \
        +trainer.save_train_generations=False \
        +trainer.test_only=True \
        trainer.resume_mode=${RESUME_CKPT_DIR:-disable} \
        trainer.resume_from_path=${RESUME_FROM_PATH:-False} \
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
        hydra.run.dir=$HYDRA_RUN_DIR \
        ${EXTRA_HYDRA_OVERRIDES:-} >> "$LOG_FILE" 2>&1

RUN_RC=$?
RUN_END_EPOCH=$(date +%s)
RUN_END_ISO=$(date -Iseconds)
RUN_WALL_SEC=$((RUN_END_EPOCH - RUN_START_EPOCH))

# Native test-path timings (patched into ray_trainer._test on 2026-04-20).
# Fall back to fit-loop training-path timings if we happen to run a fit job instead.
test_timing_s_total=$(rg -o 'test/timing_s/total:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_timing_s_gen=$(rg -o 'test/timing_s/gen:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_timing_s_reward=$(rg -o 'test/timing_s/reward:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_timing_per_token_ms_gen=$(rg -o 'test/timing_per_token_ms/gen:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_timing_per_token_ms_total=$(rg -o 'test/timing_per_token_ms/total:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_sec_per_conv=$(rg -o 'test/perf/sec_per_conv:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_gen_sec_per_conv=$(rg -o 'test/perf/gen_sec_per_conv:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_total_completion_tokens=$(rg -o 'test/perf/total_completion_tokens:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_num_finished=$(rg -o 'test/perf/num_finished_convs:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)

# Legacy fit-loop timers (only populated if a fit() step ran, not for test_only)
timing_gen=$(rg -o 'timing_s/gen:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
timing_testing=$(rg -o 'timing_s/testing:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
throughput=$(rg -o 'perf/throughput:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_acc=$(rg -o 'test/acc/locomo:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_bleu=$(rg -o 'test/bleu/locomo:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)
test_mhop=$(rg -o 'test/multi_hop_f1:[0-9.eE+-]+' "$LOG_FILE" | tail -n 1 | cut -d: -f2)

echo "[test-eval] END_TS=$RUN_END_ISO RC=$RUN_RC WALL_SEC=$RUN_WALL_SEC" | tee -a "$LOG_FILE"
echo "[test-eval] LATENCY_SUMMARY run_tag=$RUN_TAG model=$MODEL_PATH wall_sec=$RUN_WALL_SEC test_timing_s_total=${test_timing_s_total:-NA} test_timing_s_gen=${test_timing_s_gen:-NA} test_timing_per_token_ms_gen=${test_timing_per_token_ms_gen:-NA} test_sec_per_conv=${test_sec_per_conv:-NA} test_total_completion_tokens=${test_total_completion_tokens:-NA} test_acc=${test_acc:-NA} test_bleu=${test_bleu:-NA} test_mhop_f1=${test_mhop:-NA}" | tee -a "$LOG_FILE"

LAT_TSV="/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public/logs/latency_summary.tsv"
if [ ! -f "$LAT_TSV" ]; then
    echo -e "date\trun_tag\tjob_id\tmodel_path\tmax_num_turns\twall_sec\ttest_timing_s_total\ttest_timing_s_gen\ttest_timing_s_reward\ttest_timing_per_token_ms_gen\ttest_timing_per_token_ms_total\ttest_sec_per_conv\ttest_gen_sec_per_conv\ttest_total_completion_tokens\ttest_num_finished\ttiming_s_gen_fit\ttiming_s_testing_fit\tthroughput_fit\ttest_acc\ttest_bleu\ttest_mhop_f1\tlog_file\trc" > "$LAT_TSV"
fi
echo -e "$(date +%F)\t$RUN_TAG\t$JOB_ID\t$MODEL_PATH\t$MAX_NUM_TURNS\t$RUN_WALL_SEC\t${test_timing_s_total:-NA}\t${test_timing_s_gen:-NA}\t${test_timing_s_reward:-NA}\t${test_timing_per_token_ms_gen:-NA}\t${test_timing_per_token_ms_total:-NA}\t${test_sec_per_conv:-NA}\t${test_gen_sec_per_conv:-NA}\t${test_total_completion_tokens:-NA}\t${test_num_finished:-NA}\t${timing_gen:-NA}\t${timing_testing:-NA}\t${throughput:-NA}\t${test_acc:-NA}\t${test_bleu:-NA}\t${test_mhop:-NA}\t$LOG_FILE\t$RUN_RC" >> "$LAT_TSV"

echo "[test-eval] Done. Log: $LOG_FILE"
exit "$RUN_RC"
