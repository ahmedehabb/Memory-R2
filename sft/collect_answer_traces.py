#!/usr/bin/env python3
"""
collect_answer_traces.py

Runs the full ReMA pipeline (fact extraction → memory operations → QA answering)
using a strong API model on training conversations, then saves (qa_prompt → answer)
pairs as SFT data for the answer agent.

Pipeline per session chunk:
  1. Fact agent (MEMORY_REASONER_PROMPT):
       user: "Analyze ONLY the following new dialogue turns..."
       → {"facts": [...]}

  2. Memory agent (MEMORY_EXECUTOR_PROMPT + memory_v2.txt):
       user: memory_v2.txt filled with {new facts + related existing memories}
       → {"operations": [...]}
       → execute INSERT / UPDATE / DELETE on Memory object

  3. QA agent (qa.txt):
       At final session (always) + inner sessions (with --inner_qa_prob).
       For each QA pair, fill qa.txt with speaker memories → call model → save.

Output JSONL (one line per QA pair):
  {
    "conv_id": "conv-43",
    "session_id": 3,
    "chunk_id": 3,
    "question": "...",
    "ground_truth": "...",
    "messages": [
      {"role": "user", "content": "<filled qa.txt>"},
      {"role": "assistant", "content": "<answer>MODEL_ANSWER</answer>"}
    ],
    "model": "gpt-4o",
    "fact_responses": [...],     # raw fact agent outputs per turn
    "memory_responses": [...]    # raw memory agent outputs per turn
  }
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project packages are importable.
# verl/__init__.py imports tensordict (a heavy ML dep not needed here).
# Stub it out before any verl import so only the memory submodules are loaded.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (sft/ is one level down)
VERL_SRC = PROJECT_ROOT / "src" / "verl"
if str(VERL_SRC) not in sys.path:
    sys.path.insert(0, str(VERL_SRC))

import types as _types

def _stub_module(name: str) -> None:
    """Insert an empty module stub so imports don't fail for unused heavy deps."""
    if name not in sys.modules:
        sys.modules[name] = _types.ModuleType(name)

# tensordict is imported at package level in verl/__init__.py but unused here
_stub_module("tensordict")
# DataProto and protocol are re-exported from verl/__init__.py
_stub_module("verl.protocol")
_stub_module("verl.utils.logging_utils")
_stub_module("verl.single_controller")

# Provide a no-op set_basic_config so verl/__init__.py doesn't crash
sys.modules["verl.utils.logging_utils"].set_basic_config = lambda **kw: None  # type: ignore[attr-defined]
sys.modules["verl.protocol"].DataProto = object  # type: ignore[attr-defined]

from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.memory_core.prompt_generator import (
    format_turns_for_prompt,
    generate_memory_prompt_using_facts,
)
from verl.rema_trainer.memory.utils.parse_response import extract_llm_json_from_response
from verl.rema_trainer.memory.utils.qa_prompt_generator import generate_qa_prompt

# ---------------------------------------------------------------------------
# System prompts (from prompt/math/multi_turn_mamrp.py)
# ---------------------------------------------------------------------------
PROMPTS_PATH = PROJECT_ROOT / "prompt" / "math" / "multi_turn_mamrp.py"
_prompt_ns: Dict[str, Any] = {}
exec(compile(PROMPTS_PATH.read_text(), str(PROMPTS_PATH), "exec"), _prompt_ns)

FACT_SYSTEM_PROMPT: str = _prompt_ns["MEMORY_REASONER_PROMPT"]
MEMORY_SYSTEM_PROMPT: str = _prompt_ns["MEMORY_EXECUTOR_PROMPT"]

# ---------------------------------------------------------------------------
# Data preprocessing helpers (reuse from data_preprocess.py)
# ---------------------------------------------------------------------------
DATA_PREPROCESS_PATH = PROJECT_ROOT / "data" / "locomo" / "data_preprocess.py"
_dp_ns: Dict[str, Any] = {}
exec(compile(DATA_PREPROCESS_PATH.read_text(), str(DATA_PREPROCESS_PATH), "exec"), _dp_ns)

flatten_conversation = _dp_ns["flatten_conversation"]
chunk_conversation_by_session = _dp_ns["chunk_conversation_by_session"]
categorize_qas_by_recency = _dp_ns["categorize_qas_by_recency"]
max_tuple_in_chunk = _dp_ns["max_tuple_in_chunk"]
min_tuple_in_chunk = _dp_ns["min_tuple_in_chunk"]


# ---------------------------------------------------------------------------
# F1 scoring (identical to rema.py)
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    import string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call_openai(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    retries: int = 5,
    backoff: float = 5.0,
) -> str:
    """Call OpenAI-compatible chat completion, return assistant text."""
    import openai

    client = openai.OpenAI()
    # GPT-5+ / o-series: use max_completion_tokens (not max_tokens), no temperature,
    # and need a much larger budget because reasoning tokens are consumed internally.
    _NEW_PARAM_PREFIXES = ("gpt-5", "o1", "o3", "o4")
    is_new = any(model.startswith(p) for p in _NEW_PARAM_PREFIXES)
    effective_max = max_tokens * 8 if is_new else max_tokens  # 8x headroom for reasoning tokens
    extra_kwargs: Dict[str, Any] = {"max_completion_tokens": effective_max} if is_new else {"max_tokens": max_tokens, "temperature": temperature}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  [API] {exc!r} — retrying in {wait:.1f}s ({attempt+1}/{retries})")
            time.sleep(wait)
    return ""


def call_model(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _call_openai(messages, model=model, temperature=temperature, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Fact extraction prompt (mirrors generate_fact_prompts in vllm_rollout_spmd.py)
# ---------------------------------------------------------------------------

def build_fact_user_prompt(turns_data: List[Dict]) -> str:
    """Build the user-side fact extraction prompt.

    Mirrors generate_fact_prompts() in vllm_rollout_spmd.py exactly:
    format_turns_for_prompt() returns a list of dicts, which is passed
    directly into .format(turns=...) — so the template receives str(list).
    """
    prompt_template = (
        "Analyze ONLY the following new dialogue turns and extract new stable facts.\n"
        "The turns are speaker-tagged and already formatted.\n"
        "New turns:\n"
        "```\n{turns}\n```"
    )
    formatted_turns = format_turns_for_prompt(turns_data)
    return prompt_template.format(turns=formatted_turns)


# ---------------------------------------------------------------------------
# Session chunk processing
# ---------------------------------------------------------------------------

def process_session_chunk(
    turns: List[Dict],
    memory: Memory,
    manager: MemoryManager,
    model: str,
    max_turns: int,
    top_k_memories: int,
    similarity_threshold: float,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Run fact + memory agents over a session chunk (possibly multiple inner turns).

    Returns:
        fact_responses: list of raw model outputs from the fact agent (one per inner turn)
        memory_responses: list of raw model outputs from the memory agent (one per inner turn)
    """
    total_turns = len(turns)
    fact_responses: List[str] = []
    memory_responses: List[str] = []

    for i_turn in range(max_turns):
        # Slice dialogue turns for this inner turn (ceiling-division chunks)
        chunk_size = (total_turns + max_turns - 1) // max_turns
        start_idx = i_turn * chunk_size
        end_idx = min(start_idx + chunk_size, total_turns)

        if start_idx >= total_turns:
            # No more turns to process
            break

        turn_slice = turns[start_idx:end_idx]

        # ── Fact agent ──────────────────────────────────────────────────────
        fact_user = build_fact_user_prompt(turn_slice)
        if verbose:
            print(f"    [Turn {i_turn+1}/{max_turns}] Fact agent → {len(turn_slice)} dialogue turns")

        fact_raw = call_model(
            system=FACT_SYSTEM_PROMPT,
            user=fact_user,
            model=model,
            max_tokens=1024,
        )
        fact_responses.append(fact_raw)

        # Parse facts
        facts_data = extract_llm_json_from_response(fact_raw)
        if not facts_data.get("_parse_success", False):
            facts_data = {"facts": []}
        else:
            facts_data.pop("_parse_success", None)

        if verbose:
            n_facts = len(facts_data.get("facts", []))
            print(f"    [Turn {i_turn+1}/{max_turns}] Extracted {n_facts} facts")

        # ── Memory agent ─────────────────────────────────────────────────────
        memory_user = generate_memory_prompt_using_facts(
            memory,
            facts_data,
            top_k_memories_for_operations=top_k_memories,
            similarity_threshold=similarity_threshold,
            use_similarity=True,
        )
        if verbose:
            print(f"    [Turn {i_turn+1}/{max_turns}] Memory agent → building operations prompt")

        memory_raw = call_model(
            system=MEMORY_SYSTEM_PROMPT,
            user=memory_user,
            model=model,
            max_tokens=2048,
        )
        memory_responses.append(memory_raw)

        # Parse and execute memory operations
        ops_data = extract_llm_json_from_response(memory_raw)
        ops_parse_ok = ops_data.pop("_parse_success", False)
        operations = ops_data.get("operations", []) if ops_parse_ok else []

        # Attach session metadata (sample_id, session_id, session_time) to INSERT ops
        operations = manager.attach_turn_metadata_to_operations(
            operations, turns, turns[0].get("sample_id", "unknown")
        )
        result = manager.execute_batch(memory, operations)

        if verbose:
            print(
                f"    [Turn {i_turn+1}/{max_turns}] Memory ops: "
                f"{result.get('insert_successful', 0)} inserts, "
                f"{result.get('update_successful', 0)} updates, "
                f"{result.get('delete_successful', 0)} deletes "
                f"(total_memories={len(memory.memories)})"
            )

    return fact_responses, memory_responses


# ---------------------------------------------------------------------------
# QA answering
# ---------------------------------------------------------------------------

def answer_question(
    memory: Memory,
    speakers: List[str],
    question: str,
    model: str,
    top_k_per_speaker: int = 30,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True,
) -> Tuple[str, str]:
    """
    Build the QA prompt and call the model.
    Returns (qa_prompt, raw_model_response).
    """
    prompt, _, _ = generate_qa_prompt(
        memory=memory,
        speaker_1=speakers[0],
        speaker_2=speakers[1],
        question=question,
        top_k_per_speaker=top_k_per_speaker,
        similarity_threshold=similarity_threshold,
        use_similarity=use_similarity,
    )
    # qa.txt is a standalone prompt (no separate system prompt needed).
    # Use 1024 tokens — enough for chain-of-thought + <answer> tag.
    messages = [{"role": "user", "content": prompt}]
    response = _call_openai(messages, model=model, temperature=0.0, max_tokens=1024)
    return prompt, response


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(
    data_path: str,
    conv_ids: List[str],
    max_sessions: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Load and chunk conversations from locomo10.json.
    Returns {conv_id: {"speakers": [...], "chunks": [chunk_dict, ...]}}
    where each chunk_dict has keys: session_id, chunk_id, turns, qa_pairs
    """
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    conv_map = {conv.get("sample_id", f"conv-{i}"): conv for i, conv in enumerate(data)}

    result = {}
    for conv_id in conv_ids:
        if conv_id not in conv_map:
            print(f"  WARNING: conv_id '{conv_id}' not found in data, skipping.")
            continue

        conv = conv_map[conv_id]
        conversation = conv.get("conversation", {})
        qa_list = conv.get("qa", [])
        turns = flatten_conversation(conversation)

        # Identify speakers (take the first 2)
        speaker_set: List[str] = []
        for t in turns:
            spk = t.get("speaker")
            if spk and spk not in speaker_set:
                speaker_set.append(spk)
            if len(speaker_set) == 2:
                break

        # Inject sample_id into each turn (needed for manager.attach_turn_metadata_to_operations)
        for t in turns:
            t["sample_id"] = conv_id

        # Build session chunks
        chunks = []
        chunk_counter = 1
        accumulated_qas: List[Dict] = []  # cumulative QAs (CUMULATIVE_QAS_FOR_TRAIN=True)

        for chunk_turns, _offset, session_id in chunk_conversation_by_session(turns):
            if max_sessions is not None and session_id > max_sessions:
                break

            chunk_max = max_tuple_in_chunk(chunk_turns)
            chunk_min = min_tuple_in_chunk(chunk_turns)
            qa_categorized = categorize_qas_by_recency(qa_list, chunk_max, chunk_min)

            # Only current QAs for this session (mirrors USE_ONLY_CURRENT_QAS=True)
            session_qas = qa_categorized["current"][:]

            # Accumulate (mirrors CUMULATIVE_QAS_FOR_TRAIN=True)
            accumulated_qas.extend(session_qas)

            session_time = chunk_turns[0].get("session_time") if chunk_turns else None

            chunks.append(
                {
                    "conv_id": conv_id,
                    "chunk_id": chunk_counter,
                    "session_id": session_id,
                    "session_time": session_time,
                    "turns": chunk_turns,
                    # cumulative view of QAs seen so far (used for QA eval at this chunk)
                    "qa_pairs": list(accumulated_qas),
                }
            )
            chunk_counter += 1

        result[conv_id] = {"speakers": speaker_set, "chunks": chunks}
        print(
            f"  Loaded {conv_id}: {len(speaker_set)} speakers, "
            f"{len(chunks)} session chunks, "
            f"{len(accumulated_qas)} total QAs (cumulative)"
        )

    return result


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_traces(
    data_path: str,
    output_path: str,
    conv_ids: List[str],
    model: str,
    max_sessions: Optional[int],
    max_turns_per_session: int,
    top_k_memories: int,
    similarity_threshold: float,
    top_k_qa_per_speaker: int,
    inner_qa_prob: float,
    use_similarity_for_qa: bool,
    verbose: bool,
) -> None:
    """
    Main loop: for each conversation → each session chunk → fact+memory agents.
    At the final session (always) and inner sessions (with probability inner_qa_prob),
    run QA answering and save traces.
    """
    print(f"\n=== collect_answer_traces ===")
    print(f"Model       : {model}")
    print(f"Conversations: {conv_ids}")
    print(f"Max sessions : {max_sessions or 'all'}")
    print(f"Inner QA prob: {inner_qa_prob}")
    print(f"Output       : {output_path}\n")

    print("Loading conversations...")
    convs = load_conversations(data_path, conv_ids, max_sessions)
    print()

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    total_qa_samples = 0

    # Line buffering + explicit flush keeps long runs observable and safer if preempted.
    with open(output_path, "w", encoding="utf-8", buffering=1) as out_fh:

        for conv_id, conv_data in convs.items():
            speakers = conv_data["speakers"]
            chunks = conv_data["chunks"]

            if len(speakers) < 2:
                print(f"  [SKIP] {conv_id}: fewer than 2 speakers found.")
                continue

            print(f"Processing {conv_id}  (speakers: {speakers[0]}, {speakers[1]})")

            # Fresh memory for this conversation
            memory = Memory()
            manager = MemoryManager()

            n_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                session_id = chunk["session_id"]
                chunk_id = chunk["chunk_id"]
                turns = chunk["turns"]
                qa_pairs = chunk["qa_pairs"]  # cumulative at this point
                is_final = (chunk_idx == n_chunks - 1)

                print(
                    f"  Session {session_id} / chunk {chunk_id}  "
                    f"({len(turns)} turns, {len(qa_pairs)} cumulative QAs)"
                )

                # ── Step 1 & 2: fact + memory agents ─────────────────────────
                fact_responses, memory_responses = process_session_chunk(
                    turns=turns,
                    memory=memory,
                    manager=manager,
                    model=model,
                    max_turns=max_turns_per_session,
                    top_k_memories=top_k_memories,
                    similarity_threshold=similarity_threshold,
                    verbose=verbose,
                )

                # ── Step 3: QA answering ──────────────────────────────────────
                # Always at final session; at inner sessions with probability inner_qa_prob
                do_qa = is_final or (inner_qa_prob > 0.0 and random.random() < inner_qa_prob)

                if not do_qa or not qa_pairs:
                    if verbose and not do_qa:
                        print(f"    Skipping QA for inner session {session_id}")
                    continue

                print(f"    Answering {len(qa_pairs)} QA pairs...")

                for qa in qa_pairs:
                    question = qa.get("question", "")
                    ground_truth = qa.get("answer", "")
                    if not question:
                        continue

                    qa_prompt, qa_response = answer_question(
                        memory=memory,
                        speakers=speakers,
                        question=question,
                        model=model,
                        top_k_per_speaker=top_k_qa_per_speaker,
                        similarity_threshold=similarity_threshold,
                        use_similarity=use_similarity_for_qa,
                    )

                    # Extract <answer> tag for scoring (fall back to full response)
                    ans_match = re.search(
                        r"<answer>(.*?)</answer>", qa_response, re.DOTALL | re.IGNORECASE
                    )
                    extracted_answer = ans_match.group(1).strip() if ans_match else qa_response.strip()
                    f1 = compute_f1(extracted_answer, ground_truth)

                    record = {
                        # ── SFT training data ──────────────────────────────
                        # Only `messages` is needed for LLaMA-Factory SFT.
                        "messages": [
                            {"role": "user", "content": qa_prompt},
                            {"role": "assistant", "content": qa_response},
                        ],
                        # Explicit extracted final answer from <answer>...</answer>
                        # so downstream SFT/data analysis can use it directly.
                        "extracted_answer": extracted_answer,
                        # ── Metadata (for analysis / filtering) ───────────
                        "metadata": {
                            "conv_id": conv_id,
                            "session_id": session_id,
                            "chunk_id": chunk_id,
                            "is_final_session": is_final,
                            "question": question,
                            "ground_truth": ground_truth,
                            "extracted_answer": extracted_answer,
                            "f1_score": round(f1, 4),
                            "qa_category": qa.get("category"),
                            "qa_type": qa.get("qa_type", "current"),
                            "model": model,
                            # Raw pipeline traces (fact + memory agent outputs)
                            "traces": {
                                "fact_responses": fact_responses,
                                "memory_responses": memory_responses,
                            },
                        },
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    total_qa_samples += 1

                print(f"    → Saved {len(qa_pairs)} QA samples (total so far: {total_qa_samples})")

    print(f"\nDone. Saved {total_qa_samples} QA traces to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect (qa_prompt → answer) SFT traces using a strong API model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        default=str(PROJECT_ROOT / "data" / "locomo" / "locomo10.json"),
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "sft" / "answer_traces.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--conv_ids",
        nargs="+",
        default=["conv-43", "conv-47"],
        help="Conversation IDs to process (training set by default)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (e.g. gpt-4o, gpt-4.1, gpt-4.5-preview)",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help="Maximum number of sessions per conversation (None = all)",
    )
    parser.add_argument(
        "--max_turns_per_session",
        type=int,
        default=1,
        help=(
            "Number of inner fact/memory turns per session chunk. "
            "1 = single pass over the whole session (matches training default). "
            "N > 1 = dialogue turns are split into N sub-chunks and processed sequentially."
        ),
    )
    parser.add_argument(
        "--top_k_memories",
        type=int,
        default=20,
        help="Top-K memories retrieved for memory operation prompts",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.1,
        help="Similarity threshold for memory retrieval during operations",
    )
    parser.add_argument(
        "--top_k_qa_per_speaker",
        type=int,
        default=30,
        help="Top-K memories retrieved per speaker for QA prompts",
    )
    parser.add_argument(
        "--inner_qa_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of running QA answering at intermediate (non-final) sessions. "
            "0.0 = QA only at the final session. "
            "1.0 = QA at every session (with cumulative QAs available so far)."
        ),
    )
    parser.add_argument(
        "--no_similarity_for_qa",
        action="store_true",
        help="Disable similarity-based retrieval for QA (pass ALL memories to the QA prompt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used for inner_qa_prob sampling)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-turn logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    collect_traces(
        data_path=args.data_path,
        output_path=args.output,
        conv_ids=args.conv_ids,
        model=args.model,
        max_sessions=args.max_sessions,
        max_turns_per_session=args.max_turns_per_session,
        top_k_memories=args.top_k_memories,
        similarity_threshold=args.similarity_threshold,
        top_k_qa_per_speaker=args.top_k_qa_per_speaker,
        inner_qa_prob=args.inner_qa_prob,
        use_similarity_for_qa=not args.no_similarity_for_qa,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
