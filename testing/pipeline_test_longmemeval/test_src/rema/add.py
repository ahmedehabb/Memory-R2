"""
ReMA-adapted memory add pipeline for LongMemEval.

Dataset format:
  Each item has:
    - haystack_sessions: list of sessions, each a list of {role, content, has_answer}
    - haystack_dates:    list of timestamps (one per session), e.g. "2023/04/10 (Mon) 17:50"
    - question:          the QA question
    - answer:            the gold answer

  Turns use role: "user" / "assistant" — mapped to speaker "User" / "Assistant".
  dia_id is synthesised as "D{session_idx+1}:{turn_idx+1}".

Two-stage pipeline (mirrors training):
  1. memExtractor (MEMORY_REASONER_PROMPT): extracts atomic facts from formatted turns.
  2. memAgent    (MEMORY_EXECUTOR_PROMPT):  performs INSERT/UPDATE/DELETE on the Memory object.

Memory is persisted per conversation with Memory.save() and can be loaded for search.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: make verl importable from the ReMA project root
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REMA_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_VERL_SRC = os.path.join(_REMA_ROOT, "src", "verl")
if _VERL_SRC not in sys.path:
    sys.path.insert(0, _VERL_SRC)

from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.prompt_generator import (
    format_turns_for_prompt,
    generate_memory_prompt_using_facts,
)

# Prompt templates used during training — reuse exactly.
_PROMPTS_PATH = os.path.join(_REMA_ROOT, "prompt", "math", "multi_turn_mamrp.py")
_ns: dict = {}
with open(_PROMPTS_PATH) as _f:
    exec(compile(_f.read(), _PROMPTS_PATH, "exec"), _ns)
MEMORY_REASONER_PROMPT: str = _ns["MEMORY_REASONER_PROMPT"]
MEMORY_EXECUTOR_PROMPT: str = _ns["MEMORY_EXECUTOR_PROMPT"]

load_dotenv()
_STAGE2_MAX_TOKENS = int(os.getenv("REMA_LONGMEMEVAL_STAGE2_MAX_TOKENS", "1024"))
_MAX_TURNS_PER_CHUNK = int(os.getenv("REMA_LONGMEMEVAL_MAX_TURNS_PER_CHUNK", "24"))
_MAX_NUM_TURNS = int(os.getenv("REMA_LONGMEMEVAL_MAX_NUM_TURNS", "4"))
_FORCE_REPROCESS = os.getenv("REMA_LONGMEMEVAL_FORCE_REPROCESS", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that some LLMs emit."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    # Remove leading ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # Remove trailing ```
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _resolve_model_id(url: str, fallback_model: str) -> str:
    """Resolve a concrete model id from /v1/models, else return fallback."""
    base = url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if not base.endswith("/v1"):
        base = base + "/v1"
    try:
        resp = requests.get(base + "/models", timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data and isinstance(data, list):
            model_id = data[0].get("id")
            if model_id:
                return model_id
    except Exception:
        pass
    return fallback_model


def _call_llm(url: str, model: str, system_prompt: str, user_content: str,
              max_tokens: int = 2048, temperature: float = 0.0,
              retries: int = 3, retry_delay: float = 2.0) -> str:
    """POST to a vLLM-compatible OpenAI chat completions endpoint."""
    # Ensure the URL points to the chat completions endpoint
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    ctx_re = re.compile(
        r"maximum context length is (\d+) tokens.*requested (\d+) output tokens.*contains at least (\d+) input tokens",
        re.IGNORECASE | re.DOTALL,
    )
    for attempt in range(retries):
        try:
            resp = requests.post(endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"].get("content", "")
            return content if isinstance(content, str) else (content or "")
        except Exception as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status = exc.response.status_code
                err_text = exc.response.text or ""
                m = ctx_re.search(err_text)
                if status == 400 and m:
                    max_ctx = int(m.group(1))
                    input_tokens = int(m.group(3))
                    safe_budget = max_ctx - input_tokens - 16
                    new_max_tokens = min(payload.get("max_tokens", max_tokens), safe_budget)
                    if new_max_tokens >= 64 and new_max_tokens < payload.get("max_tokens", max_tokens):
                        payload["max_tokens"] = new_max_tokens
                        print(
                            f"[rema/add] Context overflow (input~{input_tokens}, ctx={max_ctx}); "
                            f"retrying with max_tokens={new_max_tokens}"
                        )
                        time.sleep(retry_delay)
                        continue
                # Robust fallback for alias/invalid model names (e.g. "base").
                if status in (400, 404):
                    resolved = _resolve_model_id(url, payload["model"])
                    if resolved != payload["model"]:
                        print(f"[rema/add] Model '{payload['model']}' unavailable; using '{resolved}'")
                        payload["model"] = resolved
                        time.sleep(retry_delay)
                        continue
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[rema/add] LLM call failed after {retries} attempts: {exc}")
                return ""
    return ""


def _parse_json_safe(text: str) -> dict:
    """Parse JSON, tolerating code fences and minor formatting issues."""
    if not text:
        return {}
    try:
        return json.loads(_strip_code_fences(text))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Turn-format conversion: LongMemEval → ReMA
# ---------------------------------------------------------------------------

_ROLE_TO_SPEAKER = {"user": "User", "assistant": "Assistant"}


def _session_to_rema_turns(session: list, session_idx: int, session_time: str) -> list:
    """
    Convert a LongMemEval session to ReMA turn format.

    LongMemEval session turn:
        {"role": "user" | "assistant", "content": str, "has_answer": bool}

    ReMA turn (as expected by format_turns_for_prompt / MEMORY_REASONER_PROMPT):
        {"speaker": str, "text": str, "dia_id": str, "session_time": str}
    """
    turns = []
    for turn_idx, turn in enumerate(session):
        speaker = _ROLE_TO_SPEAKER.get(turn.get("role", "user"), "User")
        turns.append({
            "speaker": speaker,
            "text": turn.get("content", ""),
            "dia_id": f"D{session_idx + 1}:{turn_idx + 1}",
            "session_time": session_time,
        })
    return turns


def _chunk_turns(turns: list, max_turns_per_chunk: int, max_num_turns: int) -> list[list]:
    """Split turns into rollout-style chunks or fixed-size chunks."""
    if not turns:
        return []

    total_turns = len(turns)
    if max_num_turns > 0:
        chunk_size = (total_turns + max_num_turns - 1) // max_num_turns
        chunks = []
        for current_turn in range(max_num_turns):
            start_idx = current_turn * chunk_size
            end_idx = min(start_idx + chunk_size, total_turns)
            if start_idx >= total_turns:
                break
            chunks.append(turns[start_idx:end_idx])
        return chunks

    if max_turns_per_chunk <= 0 or total_turns <= max_turns_per_chunk:
        return [turns]
    return [turns[i:i + max_turns_per_chunk] for i in range(0, len(turns), max_turns_per_chunk)]


# ---------------------------------------------------------------------------
# Two-stage pipeline: extract facts → update memory
# ---------------------------------------------------------------------------

def _run_two_stage_pipeline(
    memory: Memory,
    turns: list,
    session_idx: int,
    session_time: str,
    sample_id: str,
    memExtractor_url: str,
    memExtractor_model: str,
    memAgent_url: str,
    memAgent_model: str,
    top_k_memories_for_operations: int = 20,
    similarity_threshold: float = 0.1,
) -> None:
    """
    Run the two-stage ReMA pipeline for one session's turns, updating `memory` in place.

    Stage 1 — fact extraction (mirrors meta-agent in training):
      Input:  JSON array of formatted turns (speaker + text + dia_id).
      System: MEMORY_REASONER_PROMPT.
      Output: {"facts": [{speaker, dia_id, fact}, ...]}

    Stage 2 — memory operations (mirrors memory-agent in training):
      Input:  JSON object {"memories": [...], "facts": [...with related_memory_ids...]}.
      System: MEMORY_EXECUTOR_PROMPT.
      Output: {"operations": [{operation, ...}, ...]}
    """
    if not turns:
        return

    # --- Stage 1: fact extraction ---
    formatted_turns = format_turns_for_prompt(turns)
    stage1_input = (
        "Analyze ONLY the following new dialogue turns and extract new stable facts.\n"
        "The turns are speaker-tagged and already formatted.\n"
        "New turns:\n"
        "```json\n"
        f"{json.dumps(formatted_turns, indent=2)}\n"
        "```"
    )
    stage1_response = _call_llm(
        memExtractor_url, memExtractor_model,
        MEMORY_REASONER_PROMPT, stage1_input
    )
    facts = _parse_json_safe(stage1_response)
    if not facts.get("facts"):
        return  # nothing to process

    # --- Stage 2: memory operations ---
    # Build the normalised {memories, facts} input that MEMORY_EXECUTOR_PROMPT expects.
    # format_memory_for_prompt_for_facts does similarity search per fact and annotates
    # each fact with related_memory_ids pointing into a flat deduplicated memories list.
    stage2_input = generate_memory_prompt_using_facts(
        memory,
        facts=facts,
        top_k_memories_for_operations=top_k_memories_for_operations,
        similarity_threshold=similarity_threshold,
        use_similarity=True,
    )
    stage2_response = _call_llm(
        memAgent_url, memAgent_model,
        MEMORY_EXECUTOR_PROMPT, stage2_input,
        max_tokens=_STAGE2_MAX_TOKENS,
    )
    ops_data = _parse_json_safe(stage2_response)
    operations = ops_data.get("operations", [])
    if not isinstance(operations, list):
        operations = []

    # --- Execute operations on the Memory object ---
    for op in operations:
        op_type = op.get("operation", "").upper()
        try:
            if op_type == "INSERT":
                speaker  = op.get("speaker", "User")
                content  = (op.get("content") or "").strip()
                dia_id   = op.get("dia_id", f"D{session_idx + 1}:0")
                if content:
                    memory.insert(
                        sample_id, session_idx + 1, session_time,
                        speaker, content, dia_id
                    )
            elif op_type == "UPDATE":
                memory_id = op.get("memory_id", "")
                content   = (op.get("content") or "").strip()
                dia_id    = op.get("dia_id", f"D{session_idx + 1}:0")
                if memory_id and content:
                    memory.update(
                        memory_id, content, dia_id,
                        session_id=session_idx + 1,
                        session_time=session_time,
                    )
            elif op_type == "DELETE":
                memory_id = op.get("memory_id", "")
                if memory_id:
                    memory.delete(memory_id)
        except Exception as exc:
            print(f"[rema/add] Error executing {op_type} op: {exc}")


# ---------------------------------------------------------------------------
# Main MemoryADD class
# ---------------------------------------------------------------------------

class MemoryADD:
    """
    Processes LongMemEval conversations through the ReMA two-stage pipeline
    and persists one Memory object per conversation to disk.

    Saved files (via Memory.save):
      <memory_store_dir>/longmemeval_item_{idx}.pkl
      <memory_store_dir>/longmemeval_item_{idx}.json
    """

    def __init__(
        self,
        data_path: str = None,
        memory_store_dir: str = "memory_store",
        memExtractor_url: str = None,
        memExtractor_model: str = None,
        memAgent_url: str = None,
        memAgent_model: str = None,
        top_k_memories_for_operations: int = 20,
        similarity_threshold: float = 0.1,
        embedding_cache_dir: str = None,
    ):
        self.data_path = data_path
        self.data: list = []
        self.memory_store_dir = memory_store_dir
        self.memExtractor_url = memExtractor_url
        self.memExtractor_model = memExtractor_model
        self.memAgent_url = memAgent_url
        self.memAgent_model = memAgent_model
        self.top_k_memories_for_operations = top_k_memories_for_operations
        self.similarity_threshold = similarity_threshold
        self.embedding_cache_dir = embedding_cache_dir

        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path) as f:
            self.data = json.load(f)

    def process_conversation(self, item: dict, idx: int) -> None:
        """Process all sessions of one conversation and persist the resulting Memory."""
        sample_id = f"longmemeval_item_{idx}"
        pkl_path = os.path.join(self.memory_store_dir, f"{sample_id}.pkl")
        if (not _FORCE_REPROCESS) and os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 250:
            print(f"[rema/add] Skipping item {idx} (valid pkl exists)")
            return
        memory = Memory(
            embedding_method="openai",
            enable_cache=True,
            cache_dir=self.embedding_cache_dir,
        )

        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates    = item.get("haystack_dates", [])

        for session_idx, session in enumerate(
            tqdm(haystack_sessions, desc=f"Item {idx} sessions", leave=False)
        ):
            session_time = haystack_dates[session_idx] if session_idx < len(haystack_dates) else ""
            turns = _session_to_rema_turns(session, session_idx, session_time)
            for turns_chunk in _chunk_turns(turns, _MAX_TURNS_PER_CHUNK, _MAX_NUM_TURNS):
                _run_two_stage_pipeline(
                    memory, turns_chunk, session_idx, session_time, sample_id,
                    self.memExtractor_url, self.memExtractor_model,
                    self.memAgent_url, self.memAgent_model,
                    top_k_memories_for_operations=self.top_k_memories_for_operations,
                    similarity_threshold=self.similarity_threshold,
                )

        memory.save(sample_id, directory=self.memory_store_dir)
        print(f"[rema/add] Saved {len(memory.memories)} memories for item {idx}")

    def process_all_conversations(self, max_workers: int = 1) -> None:
        if not self.data:
            raise ValueError("No data loaded. Set data_path and call load_data() first.")

        # Default to sequential processing (memory embeddings hit OpenAI — be mindful of rate limits)
        if max_workers == 1:
            for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Conversations"):
                self.process_conversation(item, idx)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_conversation, item, idx)
                    for idx, item in enumerate(self.data)
                ]
                for future in futures:
                    future.result()
