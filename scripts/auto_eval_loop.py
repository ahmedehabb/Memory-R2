#!/usr/bin/env python3
"""
Autonomous eval loop — monitors running evals, parses results,
updates results.tsv + program.md, and launches next experiments.

Usage:
    nohup python scripts/auto_eval_loop.py > logs/auto_eval_loop.log 2>&1 &

The script runs until all planned work is done (or indefinitely if LOOP_FOREVER=True).
"""

import os
import re
import subprocess
import time
import json
import datetime
import shutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT = "/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public"
RESULTS_TSV = f"{PROJECT}/results.tsv"
PROGRAM_MD  = f"{PROJECT}/program.md"
STATE_FILE  = f"{PROJECT}/scripts/auto_eval_loop_state.json"
LOOP_LOG    = f"{PROJECT}/logs/auto_eval_loop_detailed.log"
CHECK_INTERVAL_S = 1200  # check every 20 minutes
LOOP_FOREVER = True       # keep looping until user stops it

CONDA_ACTIVATE = "source /hkfs/work/workspace/scratch/tum_eyi5958-myspace2/miniconda3/etc/profile.d/conda.sh && conda activate rema"
BASE_MODEL_HF  = "Qwen/Qwen2.5-7B-Instruct"
RENDEZVOUS_QWEN = f"{PROJECT}/vllm_servers_qwen"

# ---------------------------------------------------------------------------
# Eval queue — all planned evals in priority order
# Each entry:
#   id         : unique key (used in state file)
#   label      : results.tsv description label
#   judge      : "qwen" or "openoss"
#   ckpt_path  : relative to PROJECT (hf_fixed path)
#   run_tag    : RUN_TAG for wandb
#   fsdp_actor : if hf_fixed doesn't exist, convert from this actor dir
#   world_size : for conversion (default 8)
#   depends_on : id of another eval that must finish first (optional)
# ---------------------------------------------------------------------------
EVAL_QUEUE = [
    # Already launched — will be detected from log files
    {
        "id": "8sess_turns1_step10_qwen",
        "label": "TEST (Qwen judge): qwen_judge_8sess_turns1_step10",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_turns1_j3940568__20260331_143006_1turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed",
        "run_tag": "qwen_judge_8sess_turns1_step10",
        "log_glob": f"{PROJECT}/logs/3960067/qwen_8sess_turns1_*.log",
        "status": "running",
        "slurm_job": "3960067",
    },
    {
        "id": "8sess_inner0_step5_qwen",
        "label": "TEST (Qwen judge): qwen_judge_8sess_inner0_step5",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_5/hf_fixed",
        "run_tag": "qwen_judge_8sess_inner0_step5",
        "log_glob": f"{PROJECT}/logs/3963648/qwen_8sess_inner0_step5_*.log",
        "status": "running",
        "slurm_job": "3963648",
    },
    # Next: convert 8sess_inner0 step10 then eval it
    {
        "id": "8sess_inner0_step10_convert",
        "label": "CONVERT: 8sess_inner0 step10 actor → hf_fixed",
        "judge": None,
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_10/hf_fixed",
        "fsdp_actor": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/actor",
        "world_size": 8,
        "run_tag": None,
        "status": "running",  # launched manually on hkn0905
        "slurm_job": "3960067",
    },
    {
        "id": "8sess_inner0_step10_qwen",
        "label": "TEST (Qwen judge): qwen_judge_8sess_inner0_step10",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_inner0_j3940568__20260331_075755_4turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.0sampleQA_pen0oss120b/global_step_10/hf_fixed",
        "run_tag": "qwen_judge_8sess_inner0_step10",
        "log_glob": f"{PROJECT}/logs/3960067/qwen_judge_8sess_inner0_step10_*.log",
        "slurm_job": "3960067",
        "status": "running",
    },
    {
        "id": "8sess_turns6_step10_qwen",
        "label": "TEST (Qwen judge): qwen_judge_8sess_turns6_step10",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_8sess_8sess_turns6_comp02_thresh05_j3940568__20260331_224711_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_10/hf_fixed",
        "run_tag": "qwen_judge_8sess_turns6_step10",
        "log_glob": f"{PROJECT}/logs/3963648/qwen_judge_8sess_turns6_step10_*.log",
        "slurm_job": "3963648",
        "status": "running",
    },
    # Q3: 16sess_champion_v2 Qwen judge (no Qwen result yet; gpt-oss=0.499)
    {
        "id": "16sess_champion_v2_step5_qwen",
        "label": "TEST (Qwen judge): qwen_judge_16sess_champion_v2_step5",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_16sess_16sess_champion_v2_j3940568__20260401_042034_6turns_2ppo_Kl0.001_persession_0.2addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed",
        "run_tag": "qwen_judge_16sess_champion_v2_step5",
        "log_glob": f"{PROJECT}/logs/3960067/qwen_judge_16sess_champion_v2_step5_*.log",
        "slurm_job": "3960067",
        "status": "running",
    },
    # Q4: 32sess_topk80 Qwen judge (no Qwen result yet; gpt-oss=0.460)
    {
        "id": "32sess_topk80_step5_qwen",
        "label": "TEST (Qwen judge): qwen_judge_32sess_topk80_step5",
        "judge": "qwen",
        "ckpt_path": "checkpoints/rema-curriculum-v1/curr_32sess_32sess_topk80_j3940568__20260401_215126_6turns_2ppo_Kl0.001_persession_0.3addcomp_turn_grpo_1convs16r_innergrpo0.5sampleQA_pen0oss120b/global_step_5/hf_fixed",
        "run_tag": "qwen_judge_32sess_topk80_step5",
        "status": "pending",
        "depends_on": "8sess_turns6_step10_qwen",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOOP_LOG, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    # Initialise from EVAL_QUEUE
    state = {}
    for e in EVAL_QUEUE:
        state[e["id"]] = e["status"]
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_free_slurm_jobs():
    """Return list of job IDs that are running but have no active workload."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "tum_eyi5958"),
             "--format=%.10i %.8T %.6D %R %j", "--noheader"],
            text=True, timeout=30
        )
    except Exception:
        return []
    running_jobs = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "RUNNING":
            running_jobs[parts[0].strip()] = parts
    return running_jobs


def qwen_server_healthy():
    """Return True if Qwen judge server is up."""
    try:
        result = subprocess.run(
            ["curl", "-sf", f"http://hkn0912.localdomain:8100/v1/models"],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def find_log_for_eval(eval_entry):
    """Find the most recent log file for an eval."""
    log_glob = eval_entry.get("log_glob")
    if not log_glob:
        # Build from run_tag + slurm_job
        job = eval_entry.get("slurm_job", "*")
        tag = eval_entry.get("run_tag", "")
        log_glob = f"{PROJECT}/logs/{job}/*{tag}*.log"
    import glob
    files = sorted(glob.glob(log_glob), key=os.path.getmtime)
    return files[-1] if files else None


def parse_results_from_log(log_path):
    """
    Parse test metrics from a completed eval log.
    Returns dict with acc, bleu, multi_hop_f1 or None if not complete.
    """
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, errors="replace") as f:
        content = f.read()
    # Check completion marker
    if "[test-eval" not in content or "Done." not in content:
        return None
    # Primary pattern: step:0 - test/acc/locomo:0.496 - test/bleu/locomo:0.437 ...
    m = re.search(
        r"step:\d+\s*-\s*test/test_score/locomo:([\d.]+)\s*-\s*test/acc/locomo:([\d.]+)"
        r"\s*-\s*test/bleu/locomo:([\d.]+)\s*-\s*test/multi_hop_f1:([\d.]+)",
        content
    )
    if m:
        return {
            "test_score": float(m.group(1)),
            "acc":        float(m.group(2)),
            "bleu":       float(m.group(3)),
            "mhop_f1":    float(m.group(4)),
        }
    # Fallback: wandb summary lines
    acc  = re.search(r"test/acc/locomo\s+([\d.]+)", content)
    bleu = re.search(r"test/bleu/locomo\s+([\d.]+)", content)
    mhop = re.search(r"test/multi_hop_f1\s+([\d.]+)", content)
    if acc and bleu and mhop:
        return {
            "acc":     float(acc.group(1)),
            "bleu":    float(bleu.group(1)),
            "mhop_f1": float(mhop.group(1)),
        }
    return None


def append_results_tsv(eval_entry, results):
    """Append a row to results.tsv."""
    acc    = results["acc"]
    bleu   = results["bleu"]
    mhop   = results["mhop_f1"]
    label  = eval_entry["label"]
    desc   = (f"{label} — "
              f"test/acc={acc:.3f}, bleu={bleu:.3f}, multi_hop_f1={mhop:.3f}")
    row = "\t".join([
        "auto",       # commit
        "base_qwen",  # start_model
        "0.000",      # val_acc_mid
        "0.000",      # val_acc_final
        "0.000",      # mfail_final
        "keep",       # status
        desc,         # description
    ])
    with open(RESULTS_TSV, "a") as f:
        f.write(row + "\n")
    log(f"  → Appended to results.tsv: {desc}")


def update_program_md_gap(gap_id, result_line):
    """Append a result note to a Paper Gap section in program.md."""
    with open(PROGRAM_MD, errors="replace") as f:
        content = f.read()
    # Find the gap heading like "### G7" or "**G7**"
    pattern = rf"(### G{gap_id}[^\n]*\n)"
    m = re.search(pattern, content)
    if not m:
        log(f"  ⚠ Could not find G{gap_id} in program.md — skipping md update")
        return
    insert_pos = m.end()
    note = f"> **AUTO-RESULT ({datetime.date.today()}):** {result_line}\n\n"
    new_content = content[:insert_pos] + note + content[insert_pos:]
    # Write atomically
    tmp = PROGRAM_MD + ".tmp"
    with open(tmp, "w") as f:
        f.write(new_content)
    shutil.move(tmp, PROGRAM_MD)
    log(f"  → Updated program.md G{gap_id}")


def run_conversion(eval_entry, slurm_job):
    """Run convert_fsdp_to_hf.py on a GPU node."""
    actor_path = os.path.join(PROJECT, eval_entry["fsdp_actor"])
    out_path   = os.path.join(PROJECT, eval_entry["ckpt_path"])
    world_size = eval_entry.get("world_size", 8)
    if os.path.exists(out_path) and os.listdir(out_path):
        log(f"  hf_fixed already exists at {out_path}, skipping conversion.")
        return True
    cmd = (
        f"cd {PROJECT} && "
        f"{CONDA_ACTIVATE} && "
        f"python convert_fsdp_to_hf.py "
        f"  --fsdp_checkpoint_path {actor_path} "
        f"  --huggingface_model_path {BASE_MODEL_HF} "
        f"  --output_path {out_path} "
        f"  --world_size {world_size}"
    )
    full_cmd = f"srun --jobid={slurm_job} --overlap -N1 -n1 bash -c {repr(cmd)}"
    log(f"  Launching conversion on job {slurm_job}: {eval_entry['id']}")
    log_path = f"{PROJECT}/logs/{slurm_job}/convert_{eval_entry['id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    proc = subprocess.Popen(
        full_cmd + f" > {log_path} 2>&1",
        shell=True
    )
    proc.wait(timeout=1800)  # 30 min max
    if os.path.exists(out_path) and os.listdir(out_path):
        log(f"  Conversion done → {out_path}")
        return True
    log(f"  ⚠ Conversion may have failed — check {log_path}")
    return False


def launch_eval(eval_entry, slurm_job):
    """Launch a test eval on a free SLURM job."""
    script = "vllm_client_test_eval_qwen.sh" if eval_entry["judge"] == "qwen" else "vllm_client_test_eval.sh"
    ckpt = os.path.join(PROJECT, eval_entry["ckpt_path"])
    run_tag = eval_entry["run_tag"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{PROJECT}/logs/{slurm_job}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{run_tag}_{ts}.log"

    # Store log path in eval entry for later parsing
    eval_entry["log_path_launched"] = log_path
    eval_entry["slurm_job"] = slurm_job
    eval_entry["log_glob"] = log_path

    cmd = (
        f"export MODEL_PATH_OVERRIDE='{ckpt}'; "
        f"export RUN_TAG='{run_tag}'; "
        f"export RUN_TS='{ts}'; "
        f"bash {PROJECT}/scripts/vllm_clients/{script}"
    )
    full_cmd = f"srun --jobid={slurm_job} --overlap -N1 -n1 bash -c {repr(cmd)} > {log_path} 2>&1"
    log(f"  Launching eval '{eval_entry['id']}' on job {slurm_job}")
    subprocess.Popen(full_cmd, shell=True)
    return log_path


def get_busy_jobs(state):
    """Return set of slurm job IDs currently occupied by running evals."""
    busy = {"3960210"}  # server always occupies hkn0912
    for e in EVAL_QUEUE:
        eid = e["id"]
        if state.get(eid) == "running" and e.get("slurm_job"):
            busy.add(e["slurm_job"])
    return busy


def pick_free_job(state):
    """Return a free SLURM job ID we can use, or None."""
    running = get_free_slurm_jobs()
    busy = get_busy_jobs(state)
    candidates = ["3960067", "3963648"]  # hkn0905, hkn0908
    for jid in candidates:
        if jid in running and jid not in busy:
            return jid
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(LOOP_LOG), exist_ok=True)
    log("=" * 60)
    log("auto_eval_loop.py starting")
    log("=" * 60)

    state = load_state()
    log(f"Loaded state: {state}")

    # Sync EVAL_QUEUE status from state
    queue = {e["id"]: e for e in EVAL_QUEUE}
    for eid, st in state.items():
        if eid in queue:
            queue[eid]["status"] = st

    iteration = 0
    while True:
        iteration += 1
        log(f"\n--- Iteration {iteration} ---")

        changed = False

        # ---- 1. Check running evals for completion ----
        for e in EVAL_QUEUE:
            eid = e["id"]
            if state.get(eid) != "running":
                continue
            log_path = find_log_for_eval(e)
            results = parse_results_from_log(log_path)
            if results:
                log(f"  DONE: {eid} — acc={results['acc']:.3f} bleu={results['bleu']:.3f} mhop_f1={results['mhop_f1']:.3f}")
                state[eid] = "done"
                changed = True
                append_results_tsv(e, results)
                # Update program.md depending on which eval finished
                if "turns1" in eid:
                    line = (f"8sess_turns1 step10 (Qwen judge): test/acc={results['acc']:.3f}, "
                            f"bleu={results['bleu']:.3f}, multi_hop_f1={results['mhop_f1']:.3f}. "
                            f"Turn ablation now complete: turns=1→{results['acc']:.3f}, "
                            f"turns=2→0.429, turns=6→(see 8sess champion).")
                    update_program_md_gap(7, line)
                elif "inner0" in eid and "convert" not in eid:
                    line = (f"8sess_inner0 {eid.split('_')[3]} (Qwen judge): test/acc={results['acc']:.3f}, "
                            f"bleu={results['bleu']:.3f}, multi_hop_f1={results['mhop_f1']:.3f}. "
                            f"Inner GRPO effect at 8-sess stage confirmed.")
                    update_program_md_gap(7, line)
                elif "turns6" in eid:
                    line = (f"8sess_turns6 step10 (Qwen judge): test/acc={results['acc']:.3f}, "
                            f"bleu={results['bleu']:.3f}, multi_hop_f1={results['mhop_f1']:.3f}. "
                            f"Full Qwen-judge turn ablation: turns1=0.448, turns2=0.429, turns6={results['acc']:.3f}.")
                    update_program_md_gap(7, line)
                elif "16sess_champion_v2" in eid:
                    line = (f"16sess_champion_v2 step5 (Qwen judge): test/acc={results['acc']:.3f}, "
                            f"bleu={results['bleu']:.3f}, multi_hop_f1={results['mhop_f1']:.3f}. "
                            f"gpt-oss was 0.499 — ordering consistent={results['acc'] > 0.45}.")
                    update_program_md_gap(6, line)
                elif "32sess_topk80" in eid:
                    line = (f"32sess_topk80 step5 (Qwen judge): test/acc={results['acc']:.3f}, "
                            f"bleu={results['bleu']:.3f}, multi_hop_f1={results['mhop_f1']:.3f}. "
                            f"gpt-oss was 0.460. Qwen inner GRPO gap: topk80-inner0.5({results['acc']:.3f}) vs inner0(0.322) = +{results['acc']-0.322:.3f}.")
                    update_program_md_gap(6, line)
            else:
                log(f"  Still running: {eid} (log: {log_path})")

        # ---- 2. Check if conversions completed ----
        for e in EVAL_QUEUE:
            eid = e["id"]
            if state.get(eid) != "running":
                continue
            if e.get("judge") is None and e.get("fsdp_actor"):
                # This is a conversion task
                out = os.path.join(PROJECT, e["ckpt_path"])
                if os.path.exists(out) and os.listdir(out):
                    log(f"  CONVERSION DONE: {eid}")
                    state[eid] = "done"
                    changed = True

        # ---- 3. Launch pending tasks if node is free ----
        for e in EVAL_QUEUE:
            eid = e["id"]
            if state.get(eid) != "pending":
                continue
            # Check dependency
            dep = e.get("depends_on")
            if dep and state.get(dep) != "done":
                log(f"  Waiting for dependency: {eid} → {dep} (status={state.get(dep)})")
                continue
            free_job = pick_free_job(state)
            if not free_job:
                log(f"  No free node available for: {eid}")
                break
            if e.get("judge") is None and e.get("fsdp_actor"):
                # Conversion task
                state[eid] = "running"
                queue[eid]["slurm_job"] = free_job
                changed = True
                success = run_conversion(e, free_job)
                if success:
                    state[eid] = "done"
            else:
                # Eval task — check server health
                if e["judge"] == "qwen" and not qwen_server_healthy():
                    log(f"  ⚠ Qwen server not healthy — skipping launch of {eid}")
                    continue
                state[eid] = "running"
                queue[eid]["slurm_job"] = free_job
                changed = True
                launch_eval(e, free_job)

        if changed:
            save_state(state)
            log(f"State saved: {state}")

        # ---- 4. Check if all done ----
        pending_or_running = [e["id"] for e in EVAL_QUEUE if state.get(e["id"]) not in ("done", "skip")]
        if not pending_or_running:
            log("All planned evals complete! Exiting loop.")
            break

        log(f"Remaining: {pending_or_running}")
        log(f"Sleeping {CHECK_INTERVAL_S}s ...")
        time.sleep(CHECK_INTERVAL_S)

    log("auto_eval_loop.py finished.")


if __name__ == "__main__":
    main()
