#!/bin/bash
# Orchestrate sequential 7B evals on allocation 4031801 against the gpt-oss-120b
# answer-agent server (on 4031800). Base already running; this waits for it, then
# runs the champion, then extracts the input-token + latency metrics from both.
set -u
REPO=/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public
cd "$REPO"
CKPT="checkpoints/rema-curriculum-v1/curr_32sess_32sess_champion_v2_j3940568__20260401_125922_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed"
EVAL_JOB=4031801

script_done() { grep -qE "LATENCY_SUMMARY|step:0 - test/test_score" "$1" 2>/dev/null; }

wait_for() {  # $1=launch_log $2=step_glob $3=label
  local i L S
  for i in $(seq 1 90); do   # up to 90 min
    L="$1"; S=$(ls -t $2 2>/dev/null | head -1)
    if grep -qE "LATENCY_SUMMARY" "$L" 2>/dev/null; then echo "[orch] $3 launcher finished"; return 0; fi
    if [ -n "$S" ] && script_done "$S"; then echo "[orch] $3 metrics emitted"; sleep 60; return 0; fi
    sleep 60
  done
  echo "[orch] $3 TIMEOUT"; return 1
}

extract() {  # $1=step_log $2=label
  echo "############### $2"
  sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "test/(perf/total_input_tokens|perf/total_completion_tokens|input_tokens/(mean|max)|completion_tokens/(mean|max)|input_tokens_per_turn/mean|completion_tokens_per_turn/mean|perf/(sec_per_conv|gen_sec_per_conv)|timing_per_token_ms/gen|timing_s/(gen|reward|total)|acc/locomo|num_turns/mean|perf/total_meta_input_tokens|perf/total_reason_input_tokens|perf/total_meta_gen_tokens|perf/total_reason_gen_tokens):[0-9.]+" | sort -u
  echo "--- input_tokens_at_turn (context growth) ---"
  sed 's/\x1b\[[0-9;]*m//g' "$1" 2>/dev/null | grep -oE "test/input_tokens_at_turn/[0-9]+:[0-9.]+" | sort -t/ -k3 -n | tail -8
}

echo "[orch] waiting for BASE..."
wait_for logs/$EVAL_JOB/launch_inputtok_base_gptoss.log "logs/$EVAL_JOB/inputtok_7b_base_gptoss_*.log" BASE

echo "[orch] launching CHAMPION on $EVAL_JOB..."
nohup srun --overlap --jobid=$EVAL_JOB -N1 -n1 bash -c "cd $REPO && export VAL_KWARGS_N=1 MAX_NUM_TURNS=6 SERVER_WAIT_TIMEOUT=900 REMA_DUMP_QA=1 RUN_TAG=inputtok_7b_champion_gptoss MODEL_PATH_OVERRIDE='$CKPT' && unset USE_OPENAI_API_JUDGE && bash scripts/vllm_clients/vllm_client_test_eval.local.sh" > logs/$EVAL_JOB/launch_inputtok_champion_gptoss.log 2>&1 &
echo "[orch] champion PID $!"
sleep 30
wait_for logs/$EVAL_JOB/launch_inputtok_champion_gptoss.log "logs/$EVAL_JOB/inputtok_7b_champion_gptoss_*.log" CHAMPION

echo ""; echo "================= RESULTS ================="
extract "$(ls -t logs/$EVAL_JOB/inputtok_7b_base_gptoss_*.log | head -1)" "BASE (7B-base, gpt-oss answer)"
extract "$(ls -t logs/$EVAL_JOB/inputtok_7b_champion_gptoss_*.log | head -1)" "CHAMPION (7B-rl, gpt-oss answer)"
echo "[orch] DONE"
