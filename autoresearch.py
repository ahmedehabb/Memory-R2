#!/usr/bin/env python3
"""
AutoResearch — fully autonomous RL experiment loop for ReMA.
Runs indefinitely: polls logs, kills losers, promotes winners,
creates new experiment ideas, connects to pending SLURM jobs.

Usage:  python autoresearch.py [--poll 300]
Status: tail -f autoresearch.log
Stop:   Ctrl+C  (state saved, safe to restart)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

# ── Workspace ──────────────────────────────────────────────────────────────────
WS           = Path("/hkfs/work/workspace/scratch/tum_eyi5958-myspace2/projects/ReMA-public")
CKPT_ROOT    = WS / "checkpoints" / "rema-curriculum-v1"
STATE_FILE   = WS / "autoresearch_state.json"
LOG_FILE     = WS / "autoresearch.log"
DECISIONS_LOG= WS / "autoresearch_decisions.log"
RESULTS_TSV  = WS / "results.tsv"
BASE_SCRIPT  = WS / "vllm_client_16r.sh"          # template for new variants

# ── Stage ladder  (sessions, train_epochs, qa_top_k) ──────────────────────────
STAGE_LADDER = [(8, 5, 30), (16, 5, 50), (32, 20, 80)]

# ── Thresholds ────────────────────────────────────────────────────────────────
COLLAPSE_MEM_FAIL   = 0.80   # kill early if mem_fail exceeds this
COLLAPSE_MIN_STEPS  = 3      # ... after this many training steps
DISCARD_MEM_FAIL    = 0.50   # at decision point, discard if above this
MIN_TIME_FOR_STAGE  = {      # minimum hours needed to attempt a stage
    0: 1.5,   # 8-sess  (~1h training + final test)
    1: 2.5,   # 16-sess (~2h training + final test)
    2: 16.0,  # 32-sess (~15h training)
}
MAX_PARALLEL = 4             # max simultaneous experiments

# ── Sessions pool ─────────────────────────────────────────────────────────────
# {session_name: job_id}  — will be updated as pending jobs become available
SESSIONS_POOL = {
    "1":    "3933468",
    "3":    "3936296",
    "4":    "3936297",
    "exp4": "3933463",
}

# ── Experiment idea queue (tried in order when a session is free) ──────────────
# Each entry: sed-style overrides applied to BASE_SCRIPT
IDEA_QUEUE = [
    # Already launched as initial scouts (tracked separately)
    # New ideas to try once a slot frees up:
    {
        "name": "16r_32rollouts",
        "description": "32 rollouts — more GRPO diversity",
        "overrides": {"num_rollouts=16": "num_rollouts=32"},
    },
    {
        "name": "16r_lr1e6",
        "description": "half LR (1e-6) — more conservative",
        "overrides": {"STAGE_LR=\"2e-6\"": "STAGE_LR=\"1e-6\""},
    },
    {
        "name": "16r_lr4e6",
        "description": "2x LR (4e-6) — more aggressive",
        "overrides": {"STAGE_LR=\"2e-6\"": "STAGE_LR=\"4e-6\""},
    },
    {
        "name": "16r_kl005",
        "description": "kl=0.005 — slightly stronger regularization",
        "overrides": {"kl_loss_coef=0.003": "kl_loss_coef=0.005"},
    },
    {
        "name": "16r_gamma095",
        "description": "gamma_turn=0.95 — mild turn-level discount",
        "overrides": {"GAMMA_TURN_LEVEL=1.0": "GAMMA_TURN_LEVEL=0.95"},
    },
    {
        "name": "16r_clip03",
        "description": "clip_ratio=0.3 — less conservative clipping",
        "overrides": {"CLIP_RATIO=0.2": "CLIP_RATIO=0.3"},
    },
    {
        "name": "16r_comp01",
        "description": "compression_penalty=0.1 — lighter compression pressure",
        "overrides": {"COMPRESSION_PENALTY=0.2": "COMPRESSION_PENALTY=0.1"},
    },
    {
        "name": "16r_epochs3",
        "description": "ppo_epochs=3 — more gradient steps per batch",
        "overrides": {"PPO_EPOCHS=2": "PPO_EPOCHS=3"},
    },
    {
        "name": "3c_16r",
        "description": "3 training convs × 16 rollouts",
        "overrides": {
            "NUM_TRAIN_CONVS=1": "NUM_TRAIN_CONVS=3",
            "NUM_TEST_CONVS=7": "NUM_TEST_CONVS=6",
        },
    },
]

# ── Log helper ────────────────────────────────────────────────────────────────
def log(msg: str, also_decisions: bool = False):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_decisions:
        with open(DECISIONS_LOG, "a") as f:
            f.write(line + "\n")


# ── SLURM helpers ─────────────────────────────────────────────────────────────
def slurm_remaining_hours(job_id: str) -> float | None:
    """Return remaining wall-time hours for a SLURM job, or None if not running."""
    try:
        r = subprocess.run(
            ["squeue", "--job", job_id, "--format=%L", "--noheader"],
            capture_output=True, text=True, timeout=10
        )
        t = r.stdout.strip()
        if not t or t in ("INVALID", "N/A"):
            return None
        # Format: [D-]HH:MM:SS
        days, hms = (t.split("-") if "-" in t else (0, t))
        h, m, s = hms.split(":")
        return int(days) * 24 + int(h) + int(m) / 60 + int(s) / 3600
    except Exception:
        return None


def get_pending_jobs() -> list[str]:
    """Return job IDs of our pending (not yet running) SLURM jobs."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "tum_eyi5958"),
             "--format=%i %T", "--noheader"],
            capture_output=True, text=True, timeout=10
        )
        return [line.split()[0] for line in r.stdout.strip().splitlines()
                if "PENDING" in line]
    except Exception:
        return []


def connect_pending_job(session: str, job_id: str) -> bool:
    """Connect a tmux session to a pending (now-running) job via srun --overlap."""
    hours = slurm_remaining_hours(job_id)
    if hours is None:
        return False  # not running yet
    log(f"  Connecting session {session} to new job {job_id} ({hours:.1f}h remaining)")
    subprocess.run(
        ["tmux", "send-keys", "-t", session,
         f"srun --jobid {job_id} --overlap --pty bash", "Enter"],
        capture_output=True
    )
    time.sleep(5)
    # cd to workspace
    subprocess.run(
        ["tmux", "send-keys", "-t", session,
         f"cd {WS} && conda activate rema", "Enter"],
        capture_output=True
    )
    time.sleep(3)
    return True


# ── Metric extraction ─────────────────────────────────────────────────────────
_STEP_RE = re.compile(r"step:(\d+) - (.+)")
_KV_RE   = re.compile(r"([A-Za-z0-9_/]+):([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

def parse_step_metrics(log_path: Path) -> dict[int, dict[str, float]]:
    if not log_path.exists():
        return {}
    result: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _STEP_RE.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        kvs  = {}
        for k, v in _KV_RE.findall(m.group(2)):
            try:
                kvs[k] = float(v)
            except ValueError:
                pass
        if kvs:
            result[step] = kvs
    return result


def find_log(job_id: str, sessions: int, prefix: str = "curr") -> Path | None:
    log_dir = WS / "logs" / str(job_id)
    pat     = str(log_dir / f"{prefix}_{sessions}sess_{job_id}_*.log")
    matches = sorted(glob.glob(pat), key=os.path.getmtime)
    return Path(matches[-1]) if matches else None


def poll_experiment(exp: dict) -> dict:
    if exp["status"] in ("killed", "complete", "crashed"):
        return exp

    job_id   = exp["job_id"]
    si       = exp["stage_idx"]
    sessions, target_epochs, _ = STAGE_LADDER[si]

    # primary log: inner tee'd log; fallback: outer redirect log
    train_log = find_log(job_id, sessions)
    outer_log = WS / "logs" / f"{exp['log_prefix']}.log"
    active_log = (train_log if (train_log and train_log.exists()) else outer_log)

    if not active_log.exists():
        return exp

    text      = active_log.read_text(errors="ignore")
    step_data = parse_step_metrics(active_log)

    # Split into val lines (have val/acc/locomo) and train lines (have train/acc)
    val_steps   = {s: d for s, d in step_data.items() if "val/acc/locomo"         in d}
    train_steps = {s: d for s, d in step_data.items() if "train/acc"              in d}

    if train_steps:
        last_s = max(train_steps)
        exp["metrics"]["steps_done"] = last_s
        exp["metrics"]["train_acc"]  = train_steps[last_s].get("train/acc")
        exp["metrics"]["mem_fail"]   = train_steps[last_s].get("memory/memory_failure_rate")
        exp["metrics"]["mem_size"]   = train_steps[last_s].get("memory/memory_size")

    if val_steps:
        best_vs = val_steps[max(val_steps)]
        exp["metrics"]["val_acc"] = best_vs.get("val/acc/locomo")

    # Crash detection
    if "Error: Could not find checkpoint" in text or "OSError" in text:
        if exp["status"] not in ("killed", "stage_done"):
            exp["status"] = "crashed"
            return exp

    # Collapse detection (early kill)
    mf = exp["metrics"].get("mem_fail") or 1.0
    sd = exp["metrics"].get("steps_done") or 0
    if mf > COLLAPSE_MEM_FAIL and sd >= COLLAPSE_MIN_STEPS:
        exp["metrics"]["collapsed"] = True

    # Stage training complete
    if "Stage Complete. Next model:" in text and exp["status"] == "running":
        exp["status"] = "testing"
        for line in text.splitlines():
            if "Stage Complete. Next model:" in line:
                exp["checkpoint"] = line.split("Stage Complete. Next model:")[-1].strip()

    # Final test complete
    test_log = find_log(job_id, sessions, prefix="TEST_curr")
    if test_log and test_log.exists():
        test_data = parse_step_metrics(test_log)
        test_steps = {s: d for s, d in test_data.items() if "test/acc/locomo" in d}
        if test_steps:
            bd = test_steps[max(test_steps)]
            exp["metrics"]["test_acc"]      = bd.get("test/acc/locomo")
            exp["metrics"]["test_mem_fail"] = bd.get("memory/memory_failure_rate")
        if "[client] Training and final testing complete." in test_log.read_text(errors="ignore"):
            exp["status"] = "stage_done"

    return exp


# ── Script creation ───────────────────────────────────────────────────────────
def create_script_from_idea(idea: dict, base_model: str,
                             sessions: int, epochs: int, qa_top_k: int) -> Path:
    """Create a new .sh script for an idea by applying overrides to BASE_SCRIPT."""
    script_name = f"vllm_client_{idea['name']}.sh"
    dst = WS / script_name
    text = BASE_SCRIPT.read_text()
    # Apply all overrides
    for old, new in idea["overrides"].items():
        text = text.replace(old, new)
    # Set stage
    text = re.sub(r"^STAGES=\(.*\)$",          f"STAGES=({sessions})",   text, flags=re.MULTILINE)
    text = re.sub(r"^EPOCHS_PER_STAGE=\(.*\)$", f"EPOCHS_PER_STAGE=({epochs})", text, flags=re.MULTILINE)
    text = re.sub(r"^QA_TOP_K_PER_STAGE=\(.*\)",f"QA_TOP_K_PER_STAGE=({qa_top_k})", text, flags=re.MULTILINE)
    # Set base model
    text = re.sub(r"^export BASE_MODEL=.*$", f"export BASE_MODEL={base_model}",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^BASE_MODEL=.*$",         f"BASE_MODEL={base_model}",
                  text, flags=re.MULTILINE)
    dst.write_text(text)
    dst.chmod(0o755)
    return dst


def update_script_for_stage(script_path: Path, sessions: int, epochs: int,
                              qa_top_k: int, base_model: str):
    text = script_path.read_text()
    text = re.sub(r"^STAGES=\(.*\)$",          f"STAGES=({sessions})",   text, flags=re.MULTILINE)
    text = re.sub(r"^EPOCHS_PER_STAGE=\(.*\)$", f"EPOCHS_PER_STAGE=({epochs})", text, flags=re.MULTILINE)
    text = re.sub(r"^QA_TOP_K_PER_STAGE=\(.*\)",f"QA_TOP_K_PER_STAGE=({qa_top_k})", text, flags=re.MULTILINE)
    text = re.sub(r"^export BASE_MODEL=.*$", f"export BASE_MODEL={base_model}",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^BASE_MODEL=.*$",         f"BASE_MODEL={base_model}",
                  text, flags=re.MULTILINE)
    script_path.write_text(text)


# ── Session management ────────────────────────────────────────────────────────
def tmux_send(session: str, cmd: str, enter: bool = True):
    args = ["tmux", "send-keys", "-t", session, cmd]
    if enter:
        args.append("Enter")
    subprocess.run(args, capture_output=True)


def kill_experiment(exp: dict, reason: str):
    log(f"  ✗ KILL  {exp['name']:22s} reason={reason}", also_decisions=True)
    tmux_send(exp["session"], "C-c", enter=False)
    time.sleep(1)
    tmux_send(exp["session"], "C-c", enter=False)
    exp["status"]      = "killed"
    exp["kill_reason"] = reason


def launch_experiment(exp: dict):
    log_name = f"{exp['log_prefix']}"
    tmux_send(exp["session"],
              f"bash {exp['script']} > logs/{log_name}.log 2>&1 &")
    log(f"  ↑ LAUNCH {exp['name']:22s} sess={exp['session']} "
        f"log=logs/{log_name}.log", also_decisions=True)
    time.sleep(2)


def promote_experiment(exp: dict, state: dict) -> bool:
    """Advance experiment to the next stage. Returns False if cannot."""
    si           = exp["stage_idx"]
    next_si      = si + 1
    if next_si >= len(STAGE_LADDER):
        exp["status"] = "complete"
        return False

    sessions, epochs, qa_top_k = STAGE_LADDER[next_si]

    # Time check
    hours = slurm_remaining_hours(exp["job_id"])
    needed = MIN_TIME_FOR_STAGE[next_si]
    if hours is not None and hours < needed:
        log(f"  ! Cannot promote {exp['name']}: only {hours:.1f}h left, need {needed}h",
            also_decisions=True)
        exp["status"] = "time_out"
        return False

    # Find checkpoint
    ckpt = exp.get("checkpoint")
    if not ckpt:
        prev_sess = STAGE_LADDER[si][0]
        job_id    = exp["job_id"]
        pat = str(CKPT_ROOT / f"curr_{prev_sess}sess_{job_id}_*" /
                  "best_checkpoint_info.txt")
        hits = glob.glob(pat)
        if hits:
            step = Path(hits[0]).read_text().strip()
            ckpt = str(Path(hits[0]).parent / f"global_step_{step}" / "hf_fixed")
            exp["checkpoint"] = ckpt
        else:
            log(f"  ! Cannot promote {exp['name']}: checkpoint not found",
                also_decisions=True)
            exp["status"] = "crashed"
            return False

    log(f"  → PROMOTE {exp['name']:22s} → {sessions}-sess "
        f"(epochs={epochs}, qa_top_k={qa_top_k})", also_decisions=True)

    update_script_for_stage(WS / exp["script"], sessions, epochs, qa_top_k, ckpt)

    log_label = f"stage{sessions}_{exp['name']}"
    exp["log_prefix"] = log_label
    exp["stage_idx"]  = next_si
    exp["status"]     = "running"
    exp["metrics"]    = {k: None for k in exp["metrics"]}
    exp["metrics"]["steps_done"] = 0
    exp["metrics"]["collapsed"]  = False
    exp["checkpoint"] = None

    launch_experiment(exp)
    return True


# ── Decision logic ────────────────────────────────────────────────────────────
def score_experiment(exp: dict) -> float:
    """Score for ranking: prefer test_acc, fallback val_acc. 0 if collapsed."""
    if exp["metrics"].get("collapsed"):
        return 0.0
    if (exp["metrics"].get("mem_fail") or 1.0) > DISCARD_MEM_FAIL:
        return 0.0
    test = exp["metrics"].get("test_acc")
    val  = exp["metrics"].get("val_acc")
    return float(test or val or 0.0)


def make_stage_decision(state: dict, si: int):
    """Called when all experiments at stage si are terminal."""
    sessions = STAGE_LADDER[si][0]
    log(f"\n{'='*65}", also_decisions=True)
    log(f"DECISION  stage={sessions}-sess", also_decisions=True)

    exps = state["experiments"]
    done = {n: e for n, e in exps.items()
            if e["stage_idx"] == si and e["status"] == "stage_done"}

    log(f"  Candidates: {list(done.keys())}")
    for n, e in done.items():
        m = e["metrics"]
        log(f"  {n:22s}  val={m.get('val_acc') or 0:.3f}  "
            f"test={m.get('test_acc') or 0:.3f}  "
            f"mem_fail={m.get('mem_fail') or 1:.3f}  "
            f"steps={m.get('steps_done')}")

    scored = sorted(done.items(), key=lambda kv: score_experiment(kv[1]), reverse=True)

    # How many sessions are available for next stage
    free_sessions = [s for s, j in state["sessions_pool"].items()
                     if not any(e["session"] == s and e["status"] == "running"
                                for e in exps.values())]

    next_si = si + 1

    # ── Promote winner(s) to next stage ──────────────────────────────────────
    n_promote = min(len(scored), len(free_sessions), 2 if next_si < 2 else 1)
    promoted  = 0
    for name, exp in scored:
        s = score_experiment(exp)
        if s <= 0:
            log(f"  ✗ DISCARD {name} (score=0, collapse/mem_fail)", also_decisions=True)
            exp["status"] = "killed"
            continue
        if promoted < n_promote and next_si < len(STAGE_LADDER):
            if promote_experiment(exp, state):
                promoted += 1
        else:
            log(f"  ✗ SKIP    {name} (score={s:.3f}, not promoted)", also_decisions=True)
            exp["status"] = "killed"

    if promoted == 0:
        log("  ! No experiments promoted. All collapsed or timed out.", also_decisions=True)

    # ── Use freed sessions to try new ideas from queue ────────────────────────
    # (only try new ideas if we're not yet at the final stage)
    if next_si < len(STAGE_LADDER) - 1:  # don't branch at stage 32
        tried_names = {e["name"] for e in exps.values()}
        untried     = [idea for idea in state["idea_queue"]
                       if idea["name"] not in tried_names]

        # Re-evaluate which sessions are now free
        busy_sessions = {e["session"] for e in exps.values()
                         if e["status"] in ("running", "testing")}
        free_now = [s for s, j in state["sessions_pool"].items()
                    if s not in busy_sessions]

        for free_s in free_now:
            if not untried:
                break
            idea   = untried.pop(0)
            job_id = state["sessions_pool"][free_s]
            hours  = slurm_remaining_hours(job_id)
            if hours and hours < MIN_TIME_FOR_STAGE[0]:
                log(f"  ! Skip idea {idea['name']}: session {free_s} only {hours:.1f}h left")
                continue

            # Remove this idea from the queue
            state["idea_queue"] = [i for i in state["idea_queue"]
                                    if i["name"] != idea["name"]]

            s0, e0, q0 = STAGE_LADDER[0]
            ckpt = _best_checkpoint(state)
            script_path = create_script_from_idea(idea, ckpt, s0, e0, q0)

            new_exp = {
                "name":        idea["name"],
                "description": idea["description"],
                "session":     free_s,
                "job_id":      job_id,
                "script":      script_path.name,
                "log_prefix":  f"scout_{idea['name']}",
                "stage_idx":   0,
                "status":      "running",
                "metrics":     {"steps_done": 0, "collapsed": False},
                "checkpoint":  None,
            }
            exps[idea["name"]] = new_exp
            launch_experiment(new_exp)
            log(f"  ★ NEW IDEA {idea['name']:20s} → sess={free_s}: {idea['description']}",
                also_decisions=True)


def _best_checkpoint(state: dict) -> str:
    """Return the best available checkpoint (highest test_acc stage_done exp)."""
    best_exp = max(
        (e for e in state["experiments"].values()
         if e["status"] in ("stage_done", "complete") and e.get("checkpoint")),
        key=score_experiment,
        default=None
    )
    if best_exp and best_exp.get("checkpoint"):
        return best_exp["checkpoint"]
    # fallback: original starting checkpoint from BASE_SCRIPT
    text = BASE_SCRIPT.read_text()
    for line in text.splitlines():
        if line.startswith("export BASE_MODEL=") or line.startswith("BASE_MODEL="):
            return line.split("=", 1)[1].strip()
    return ""


# ── Pending job monitor ───────────────────────────────────────────────────────
def check_for_new_jobs(state: dict):
    """If pending jobs started, add them to sessions_pool and assign free ideas."""
    pending = get_pending_jobs()
    # Find running jobs we don't know about yet (pending→running transition)
    known_jobs = set(state["sessions_pool"].values())
    r = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "tum_eyi5958"),
         "--format=%i %T %N", "--noheader"],
        capture_output=True, text=True, timeout=10
    )
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "RUNNING" and parts[0] not in known_jobs:
            job_id = parts[0]
            # Find a free tmux session or create session name
            used_sessions = set(state["sessions_pool"].keys())
            for candidate in [str(i) for i in range(6, 20)] + [f"new{job_id}"] :
                if candidate not in used_sessions:
                    log(f"  + New running job {job_id} → creating tmux session {candidate}",
                        also_decisions=True)
                    # Create new tmux session and connect to job
                    subprocess.run(["tmux", "new-session", "-d", "-s", candidate],
                                    capture_output=True)
                    time.sleep(1)
                    tmux_send(candidate, f"cd {WS}")
                    time.sleep(1)
                    if connect_pending_job(candidate, job_id):
                        state["sessions_pool"][candidate] = job_id
                    break


# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        # patch idea_queue if missing
        if "idea_queue" not in s:
            s["idea_queue"] = IDEA_QUEUE
        if "sessions_pool" not in s:
            s["sessions_pool"] = SESSIONS_POOL
        return s

    return {
        "sessions_pool": dict(SESSIONS_POOL),
        "idea_queue": list(IDEA_QUEUE),
        "experiments": {
            "16r": {
                "name": "16r", "description": "16r baseline",
                "session": "1", "job_id": "3933468",
                "script": "vllm_client_16r.sh", "log_prefix": "scout_16r",
                "stage_idx": 0, "status": "running",
                "metrics": {"steps_done": 0, "collapsed": False},
                "checkpoint": None,
            },
            "16r_gamma09": {
                "name": "16r_gamma09", "description": "16r+gamma0.9",
                "session": "3", "job_id": "3936296",
                "script": "vllm_client_16r_gamma09.sh", "log_prefix": "scout_16r_gamma09",
                "stage_idx": 0, "status": "running",
                "metrics": {"steps_done": 0, "collapsed": False},
                "checkpoint": None,
            },
            "16r_kl001": {
                "name": "16r_kl001", "description": "16r+kl0.001",
                "session": "4", "job_id": "3936297",
                "script": "vllm_client_16r_kl001.sh", "log_prefix": "scout_16r_kl001",
                "stage_idx": 0, "status": "running",
                "metrics": {"steps_done": 0, "collapsed": False},
                "checkpoint": None,
            },
            "2c_16r": {
                "name": "2c_16r", "description": "2c×16r",
                "session": "exp4", "job_id": "3933463",
                "script": "vllm_client_2c16r.sh", "log_prefix": "scout_2c16r",
                "stage_idx": 0, "status": "running",
                "metrics": {"steps_done": 0, "collapsed": False},
                "checkpoint": None,
            },
        },
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def update_results_tsv(state: dict):
    lines = ["name\tstage\tsteps\ttrain_acc\tval_acc\ttest_acc\tmem_fail\tstatus\tdescription"]
    for name, exp in sorted(state["experiments"].items()):
        m = exp["metrics"]
        sess = STAGE_LADDER[exp["stage_idx"]][0]
        lines.append(
            f"{name}\t{sess}s\t{m.get('steps_done') or '-'}\t"
            f"{m.get('train_acc') or '-'}\t{m.get('val_acc') or '-'}\t"
            f"{m.get('test_acc') or '-'}\t{m.get('mem_fail') or '-'}\t"
            f"{exp['status']}\t{exp['description']}"
        )
    RESULTS_TSV.write_text("\n".join(lines) + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
def all_terminal_at_stage(state: dict, si: int) -> bool:
    at_stage = [e for e in state["experiments"].values() if e["stage_idx"] == si]
    if not at_stage:
        return False
    return all(e["status"] not in ("running", "testing") for e in at_stage)


def any_active(state: dict) -> bool:
    return any(e["status"] in ("running", "testing")
               for e in state["experiments"].values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=300, help="Poll seconds (default 300 = 5 min)")
    args = parser.parse_args()

    log("=" * 65, also_decisions=True)
    log("AutoResearch STARTED", also_decisions=True)
    log(f"Poll={args.poll}s  Stages={STAGE_LADDER}", also_decisions=True)
    log("=" * 65, also_decisions=True)

    os.chdir(WS)
    state = load_state()
    save_state(state)

    iteration = 0
    while True:
        iteration += 1
        log(f"\n--- Poll #{iteration} ({datetime.now().strftime('%H:%M')}) ---")

        # ── Check for newly started SLURM jobs ────────────────────────────────
        check_for_new_jobs(state)

        # ── Poll all active experiments ───────────────────────────────────────
        for name, exp in list(state["experiments"].items()):
            if exp["status"] in ("killed", "complete", "crashed", "time_out"):
                continue
            state["experiments"][name] = poll_experiment(exp)
            exp = state["experiments"][name]

            # Early kill on collapse
            if exp["metrics"].get("collapsed") and exp["status"] not in ("killed", "stage_done"):
                kill_experiment(exp, "memory collapse")

        # ── Print summary table ───────────────────────────────────────────────
        log(f"  {'name':22s} {'stg':4s} {'stp':4s} {'tr_acc':7s} "
            f"{'val_acc':8s} {'test_acc':9s} {'mem_fail':9s} status")
        for name, exp in sorted(state["experiments"].items()):
            m = exp["metrics"]
            log(f"  {name:22s} {STAGE_LADDER[exp['stage_idx']][0]:3d}s "
                f"{str(m.get('steps_done') or '?'):4s} "
                f"{m.get('train_acc') or 0:7.3f} "
                f"{m.get('val_acc') or 0:8.3f} "
                f"{m.get('test_acc') or 0:9.3f} "
                f"{m.get('mem_fail') or 1:9.3f} "
                f"{exp['status']}")

        # ── Check for stage completion & make decisions ───────────────────────
        active_stage_indices = {e["stage_idx"] for e in state["experiments"].values()
                                 if e["status"] not in ("killed", "crashed", "complete", "time_out")}
        for si in sorted(active_stage_indices):
            if all_terminal_at_stage(state, si):
                make_stage_decision(state, si)
                break

        # ── Stop if nothing left to do ────────────────────────────────────────
        if not any_active(state) and not state["idea_queue"]:
            have_pending = any(e["status"] == "stage_done"
                               for e in state["experiments"].values())
            if not have_pending:
                log("\n🎉 All experiments terminal and idea queue empty. Done.",
                    also_decisions=True)
                break

        save_state(state)
        update_results_tsv(state)
        log(f"  Sleeping {args.poll}s...")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
