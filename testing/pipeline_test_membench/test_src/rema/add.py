"""
ReMA-adapted memory add pipeline for MemBench.

Dataset format (all_sampled1.json / all_sampled20.json):
  Dict keyed by category name (Comparative, Knowledge_updating, Post_processing,
  Multi-hop, Single-hop, Emotion, Preference). Each value is a list of items:
    - tid:          unique conversation id
    - message_list: list of sessions, each a list of turn dicts:
        Most categories: {sid, user_message, assistant_message, time, place}
        Emotion/Preference: {mid, user, assistant, time, place}
    - QA:           dict {question, answer, ground_truth, choices, ...}

  Each turn pair (user + assistant) is expanded into two ReMA turns:
    speaker "User"      → {speaker: "User",      text: user_message,      dia_id: "D{s+1}:{t*2+1}"}
    speaker "Assistant" → {speaker: "Assistant",  text: assistant_message, dia_id: "D{s+1}:{t*2+2}"}
  session_time is taken from the first turn in each session.

Two-stage pipeline (mirrors training):
  1. memExtractor (MEMORY_REASONER_PROMPT): extracts atomic facts.
  2. memAgent    (MEMORY_EXECUTOR_PROMPT):  INSERT/UPDATE/DELETE on Memory.

Memory is persisted per item with Memory.save():
  <memory_store_dir>/membench_{category}_{tid}.pkl / .json
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
# Bootstrap
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

_PROMPTS_PATH = os.path.join(_REMA_ROOT, "prompt", "math", "multi_turn_mamrp.py")
_ns: dict = {}
with open(_PROMPTS_PATH) as _f:
    exec(compile(_f.read(), _PROMPTS_PATH, "exec"), _ns)
MEMORY_REASONER_PROMPT: str = _ns["MEMORY_REASONER_PROMPT"]
MEMORY_EXECUTOR_PROMPT: str = _ns["MEMORY_EXECUTOR_PROMPT"]

load_dotenv()
_STAGE2_MAX_TOKENS = int(os.getenv("REMA_MEMBENCH_STAGE2_MAX_TOKENS", "1024"))
_MAX_TURNS_PER_CHUNK = int(os.getenv("REMA_MEMBENCH_MAX_TURNS_PER_CHUNK", "24"))
_MAX_NUM_TURNS = int(os.getenv("REMA_MEMBENCH_MAX_NUM_TURNS", "4"))
_FORCE_REPROCESS = os.getenv("REMA_MEMBENCH_FORCE_REPROCESS", "0") == "1"

# Categories using different field names for turns
_EMOTION_PREF_CATEGORIES = {"Emotion", "Preference"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
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
    # vLLM returns 400 when prompt_tokens + max_tokens exceeds model context.
    # We adapt max_tokens downward and retry instead of silently failing to 0 memories.
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
            # Handle context-length 400s by shrinking generation budget.
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status = exc.response.status_code
                err_text = exc.response.text or ""
                m = ctx_re.search(err_text)
                if status == 400 and m:
                    max_ctx = int(m.group(1))
                    input_tokens = int(m.group(3))
                    # Keep a small safety margin for tokenizer accounting.
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
    if not text:
        return {}
    try:
        return json.loads(_strip_code_fences(text))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Turn-format conversion: MemBench → ReMA
# ---------------------------------------------------------------------------

def _session_to_rema_turns(session: list, session_idx: int, category: str) -> list:
    """
    Convert a MemBench session to ReMA turn format.

    Each MemBench turn is a user-assistant pair → expanded into two ReMA turns
    (one for "User", one for "Assistant") so that the fact extractor sees a proper
    back-and-forth dialogue matching the LoCoMo training format.

    dia_id encodes both session and turn position:
      "D{session_idx+1}:{turn_idx*2+1}"  ← User turn
      "D{session_idx+1}:{turn_idx*2+2}"  ← Assistant turn

    session_time is taken from the first turn's "time" field, matching how LoCoMo
    stores a single timestamp per session (session_N_date_time).
    """
    turns = []
    is_emotion_pref = category in _EMOTION_PREF_CATEGORIES
    # session_time from the first turn in the session
    session_time = session[0].get("time", "") if session else ""

    for turn_idx, turn in enumerate(session):
        if is_emotion_pref:
            user_text      = turn.get("user", "")
            assistant_text = turn.get("assistant", "")
        else:
            user_text      = turn.get("user_message", "")
            assistant_text = turn.get("assistant_message", "")

        # User turn
        if user_text:
            turns.append({
                "speaker": "User",
                "text": user_text,
                "dia_id": f"D{session_idx + 1}:{turn_idx * 2 + 1}",
                "session_time": session_time,
            })
        # Assistant turn (the assistant's response is context the model saw)
        if assistant_text:
            turns.append({
                "speaker": "Assistant",
                "text": assistant_text,
                "dia_id": f"D{session_idx + 1}:{turn_idx * 2 + 2}",
                "session_time": session_time,
            })

    return turns


def _chunk_turns(turns: list, max_turns_per_chunk: int, max_num_turns: int) -> list[list]:
    """
    Split a session's turns into sequential chunks.

    Priority:
    1) If max_num_turns > 0, use rollout-style chunking:
       chunk_size = ceil(total_turns / max_num_turns)
    2) Else if max_turns_per_chunk > 0, use fixed-size chunking.
    3) Else return the full turn list.
    """
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
# Two-stage pipeline
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
    if not turns:
        return

    # Stage 1: fact extraction
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
        MEMORY_REASONER_PROMPT, stage1_input,
    )
    facts = _parse_json_safe(stage1_response)
    if not facts.get("facts"):
        return

    # Stage 2: memory operations
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

    for op in operations:
        op_type = op.get("operation", "").upper()
        try:
            if op_type == "INSERT":
                speaker = op.get("speaker", "User")
                content = (op.get("content") or "").strip()
                dia_id  = op.get("dia_id", f"D{session_idx + 1}:0")
                if content:
                    memory.insert(
                        sample_id, session_idx + 1, session_time,
                        speaker, content, dia_id,
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
    Processes MemBench conversations through the ReMA two-stage pipeline.

    One Memory object per (category, tid) pair, persisted to:
      <memory_store_dir>/membench_{category}_{tid}.pkl / .json
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
        self.data: dict = {}
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

    def process_item(self, item: dict, category: str) -> None:
        """Process all sessions of one MemBench item and persist Memory."""
        tid = item.get("tid", "unknown")
        sample_id = f"membench_{category}_{tid}"
        pkl_path = os.path.join(self.memory_store_dir, f"{sample_id}.pkl")
        if (not _FORCE_REPROCESS) and os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 250:
            print(f"[rema/add] Skipping {sample_id} (valid pkl exists)")
            return
        memory = Memory(
            embedding_method="openai",
            enable_cache=True,
            cache_dir=self.embedding_cache_dir,
        )

        message_list = item.get("message_list", [])

        for session_idx, session in enumerate(
            tqdm(message_list, desc=f"{category}/{tid} sessions", leave=False)
        ):
            session_time = session[0].get("time", "") if session else ""
            turns = _session_to_rema_turns(session, session_idx, category)
            for turns_chunk in _chunk_turns(turns, _MAX_TURNS_PER_CHUNK, _MAX_NUM_TURNS):
                _run_two_stage_pipeline(
                    memory, turns_chunk, session_idx, session_time, sample_id,
                    self.memExtractor_url, self.memExtractor_model,
                    self.memAgent_url, self.memAgent_model,
                    top_k_memories_for_operations=self.top_k_memories_for_operations,
                    similarity_threshold=self.similarity_threshold,
                )

        memory.save(sample_id, directory=self.memory_store_dir)
        print(f"[rema/add] Saved {len(memory.memories)} memories for {sample_id}")

    def process_all_conversations(self, max_workers: int = 1) -> None:
        if not self.data:
            raise ValueError("No data loaded. Set data_path and call load_data() first.")

        all_items = [
            (item, category)
            for category, items in self.data.items()
            for item in items
        ]

        if max_workers == 1:
            for item, category in tqdm(all_items, desc="Items"):
                self.process_item(item, category)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_item, item, category)
                    for item, category in all_items
                ]
                for future in futures:
                    future.result()
