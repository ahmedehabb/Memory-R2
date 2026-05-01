#!/usr/bin/env python3
"""Build RLHF parquet files for normal verl.trainer.main_ppo from answer trace JSONL.

Input JSONL rows are expected to look like data/sft/answer_traces_*.jsonl and contain:
- messages[0].content: QA prompt used as user prompt
- extracted_answer or metadata.extracted_answer: target answer text
- optional metadata fields (conv/session/chunk/f1/question)

Output parquet columns are compatible with RLHFDataset + NaiveRewardManager:
- data_source
- prompt (chat list)
- reward_model.style / reward_model.ground_truth
- question
- extra_info
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Failed to parse JSON at {path}:{ln}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            rows.append(obj)
    return rows


def _to_rlhf_row(obj: Dict[str, Any], src_name: str, index: int) -> Dict[str, Any] | None:
    md = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
    messages = obj.get("messages", []) if isinstance(obj.get("messages"), list) else []

    prompt_text = ""
    if messages and isinstance(messages[0], dict):
        prompt_text = str(messages[0].get("content", ""))

    # Prefer pre-extracted answer from trace; fallback to metadata fields.
    target = obj.get("extracted_answer")
    if not isinstance(target, str) or not target.strip():
        target = md.get("extracted_answer")
    if not isinstance(target, str) or not target.strip():
        target = md.get("ground_truth")
    if not isinstance(target, str):
        target = ""

    question = md.get("question")
    if not isinstance(question, str) or not question.strip():
        question = prompt_text

    if not prompt_text.strip() or not target.strip():
        return None

    return {
        "data_source": "rema/answer_tag_f1",
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "question": question,
        "reward_model": {
            "style": "rule",
            "ground_truth": target,
        },
        "extra_info": {
            "split": "",
            "index": index,
            "source_file": src_name,
            "conv_id": md.get("conv_id"),
            "session_id": md.get("session_id"),
            "chunk_id": md.get("chunk_id"),
            "question": question,
            "target_extracted_answer": target,
            "trace_f1": md.get("f1_score"),
        },
    }


def _write_parquet(rows: List[Dict[str, Any]], out_path: Path) -> None:
    from datasets import Dataset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert answer trace JSONL into RLHF parquet for normal trainer")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more answer_traces JSONL files",
    )
    parser.add_argument("--train-out", default="data/sft_rlhf/train.parquet")
    parser.add_argument("--val-out", default="data/sft_rlhf/val.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-trace-f1",
        type=float,
        default=None,
        help="Optional filter using metadata.f1_score from source traces",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    all_rows: List[Dict[str, Any]] = []
    dropped = 0

    for inp in args.inputs:
        p = Path(inp)
        src_rows = _read_jsonl(p)
        for i, obj in enumerate(src_rows):
            md = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
            if args.min_trace_f1 is not None:
                score = md.get("f1_score")
                if not isinstance(score, (int, float)) or float(score) < args.min_trace_f1:
                    continue

            row = _to_rlhf_row(obj, p.name, len(all_rows))
            if row is None:
                dropped += 1
                continue
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No valid rows produced. Check inputs and filters.")

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    n_total = len(all_rows)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError("Train split became empty; lower --val-ratio.")

    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train:]

    for idx, row in enumerate(train_rows):
        row["extra_info"]["split"] = "train"
        row["extra_info"]["index"] = idx
    for idx, row in enumerate(val_rows):
        row["extra_info"]["split"] = "val"
        row["extra_info"]["index"] = idx

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    _write_parquet(train_rows, train_out)
    _write_parquet(val_rows, val_out)

    print("[prepare_rlhf_from_traces] done")
    print(f"total_rows={n_total} dropped_rows={dropped}")
    print(f"train_rows={len(train_rows)} -> {train_out}")
    print(f"val_rows={len(val_rows)} -> {val_out}")


if __name__ == "__main__":
    main()
