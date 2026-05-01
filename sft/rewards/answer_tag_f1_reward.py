#!/usr/bin/env python3
"""Custom reward function for verl.trainer.main_ppo.

Reward = token-level F1 between:
- prediction: text extracted from model response inside <answer>...</answer>
- target: reward_model.ground_truth from parquet row (prepared from trace extracted_answer)
"""

from __future__ import annotations

import re
import string
from typing import Any


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, truth: str) -> float:
    p = _normalize(pred).split()
    t = _normalize(truth).split()
    if len(p) == 0 or len(t) == 0:
        return float(p == t)
    common = set(p) & set(t)
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall = len(common) / len(t)
    return 2.0 * precision * recall / (precision + recall)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None) -> float:
    _ = data_source
    _ = extra_info
    pred = _extract_answer(solution_str)
    truth = ground_truth if isinstance(ground_truth, str) else ""
    return float(_f1(pred, truth))
