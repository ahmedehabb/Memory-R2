"""
ReMA-adapted memory search pipeline for MSC.

Loads per-conversation Memory objects saved by add.py, searches for memories
relevant to each QA question, then calls answerBot.

Expected memory files:
  <memory_store_dir>/msc_item_{idx}.pkl  (fast path)
  <memory_store_dir>/msc_item_{idx}.json (fallback)
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
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

load_dotenv()

# ---------------------------------------------------------------------------
# Answer prompt — two named speakers, matching LoCoMo pipeline style
# ---------------------------------------------------------------------------
ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

You have access to memories from a multi-session conversation between two people.
These memories contain timestamped facts that may be relevant to answering the question.

Instructions:
1. Carefully analyse all provided memories.
2. Pay special attention to timestamps to understand the chronological order.
3. If the question involves relative time references, convert them to absolute dates
   using the memory timestamp as the reference point.
4. If memories are contradictory, prioritise the most recent one.
5. The answer should be concise (ideally 5–6 words or fewer).
6. Output the final answer only in this format, with no extra text: <answer>YOUR_FINAL_ANSWER</answer>

Memories:
{memories}

Question: {question}

Answer step by step, and output the final answer in this format, with no extra text: <answer>YOUR_FINAL_ANSWER</answer>
""".strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _answer_extraction(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    for pattern in [
        r"<answer>(.*?)</answer>",
        r"\*\*Answer:\*\*\s*(.*)",
        r"\*\*Answer:\s*(.*?)\*\*",
        r"Answer:\s*(.*)",
    ]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text.strip()


def _call_answerbot(url: str, model: str, prompt: str,
                    max_tokens: int = 2048, temperature: float = 0.0,
                    retries: int = 3, retry_delay: float = 2.0) -> str:
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else ""
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[rema/search] answerBot call failed: {exc}")
                return ""
    return ""


def _load_memory(memory_store_dir: str, sample_id: str,
                 embedding_cache_dir: str = None) -> Memory:
    mem = Memory(embedding_method="openai", enable_cache=True, cache_dir=embedding_cache_dir)
    pkl_path  = Path(memory_store_dir) / f"{sample_id}.pkl"
    json_path = Path(memory_store_dir) / f"{sample_id}.json"

    if pkl_path.exists():
        mem.load(sample_id, directory=memory_store_dir, format="pickle")
    elif json_path.exists():
        with open(json_path) as f:
            memories = json.load(f)
        for m in memories:
            try:
                mem.insert(
                    m.get("sample_id", sample_id),
                    m.get("session_id", 0),
                    m.get("session_time", ""),
                    m.get("speaker", ""),
                    m.get("content", ""),
                    m.get("dia_ids", [""])[0],
                )
            except Exception:
                pass
    else:
        print(f"[rema/search] No memory file found for {sample_id}")
    return mem


# ---------------------------------------------------------------------------
# MemorySearch
# ---------------------------------------------------------------------------

class MemorySearch:
    """Answers MSC QA questions by searching per-conversation ReMA Memory objects."""

    def __init__(
        self,
        output_path: str = "results/rema_msc_results.json",
        memory_store_dir: str = "memory_store",
        answerBot_url: str = None,
        answerBot_model: str = None,
        top_k: int = 30,
        similarity_threshold: float = 0.0,
        embedding_cache_dir: str = None,
    ):
        self.output_path = output_path
        self.memory_store_dir = memory_store_dir
        self.answerBot_url = answerBot_url
        self.answerBot_model = answerBot_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_cache_dir = embedding_cache_dir
        self.results: dict = defaultdict(list)

    def _search_memory(self, mem: Memory, query: str) -> tuple:
        t0 = time.time()
        results = mem.search(
            query,
            top_k=self.top_k,
            min_score=self.similarity_threshold,
            search_method="text-embedding",
        )
        elapsed = time.time() - t0
        formatted = []
        for mem_dict, _ in results:
            session_time = mem_dict.get("session_time", "")
            speaker      = mem_dict.get("speaker", "")
            content      = mem_dict.get("content", "")
            formatted.append(f"{session_time} — {speaker}: {content}")
        return formatted, elapsed

    def answer_question(self, mem: Memory, question: str) -> tuple:
        memories_list, search_time = self._search_memory(mem, question)
        memories_str = json.dumps(memories_list, indent=2)
        prompt = ANSWER_PROMPT.format(memories=memories_str, question=question)

        t0 = time.time()
        raw_response = _call_answerbot(self.answerBot_url, self.answerBot_model, prompt)
        response_time = time.time() - t0

        answer = _answer_extraction(raw_response)
        return answer, memories_list, search_time, response_time

    def process_data_file(self, file_path: str) -> None:
        with open(file_path) as f:
            data = json.load(f)

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Answering questions"):
            sample_id = f"msc_item_{idx}"
            mem = _load_memory(self.memory_store_dir, sample_id, self.embedding_cache_dir)

            qa       = item.get("qa", {})
            question = qa.get("question", "")
            answer   = qa.get("answer", "")

            pred, memories_used, search_time, response_time = self.answer_question(mem, question)

            result = {
                "question":      question,
                "answer":        answer,
                "response":      pred,
                "memories_used": memories_used,
                "num_memories":  len(memories_used),
                "search_time":   search_time,
                "response_time": response_time,
            }
            self.results[idx].append(result)

            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"[rema/search] Results saved to {self.output_path}")
