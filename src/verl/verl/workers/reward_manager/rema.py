# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import json
import random
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from typing import Dict

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ThreadPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException
from verl.rema_trainer.memory.utils.parse_response import extract_answer_from_text
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.utils.qa_prompt_generator import generate_qa_prompt
from verl.rema_trainer.memory.judge_llm import judge_with_llm
import re


def _int_env(name: str, default: int, min_value: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if min_value is not None:
        return max(min_value, parsed)
    return parsed


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _reward_speed_config() -> dict:
    strategy = os.getenv("REMA_REWARD_QA_SAMPLE_STRATEGY", "first").strip().lower()
    if strategy not in {"first", "random"}:
        strategy = "first"

    return {
        # Cap outer parallelism across samples to avoid network/API overload.
        "max_outer_workers": _int_env("REMA_REWARD_MAX_OUTER_WORKERS", 16, min_value=1),
        # Cap per-sample parallel QA judging to limit nested thread contention.
        "max_inner_workers": _int_env("REMA_REWARD_MAX_INNER_WORKERS", 4, min_value=1),
        # Phase-specific QA caps (-1: no cap, 0: all QAs, >0: cap QAs).
        "max_qa_train_inner": _int_env("REMA_REWARD_MAX_QA_TRAIN_INNER", -1),
        "max_qa_train_terminal": _int_env("REMA_REWARD_MAX_QA_TRAIN_TERMINAL", -1),
        "max_qa_eval": _int_env("REMA_REWARD_MAX_QA_EVAL", -1),
        "qa_sample_strategy": strategy,
        # If true, random QA subsampling ignores rollout_idx so all rollouts share the same QA subset.
        "same_qas_across_rollouts": _bool_env("REMA_REWARD_SAME_QAS_ACROSS_ROLLOUTS", False),
        # Optional global seed to make random QA subsampling reproducible across runs.
        "qa_sample_seed": _int_env("REMA_REWARD_QA_SAMPLE_SEED", 0),
        # QA retrieval knobs.
        "qa_top_k_per_speaker": _int_env("REMA_REWARD_QA_TOP_K_PER_SPEAKER", 30, min_value=1),
        "qa_similarity_threshold": _float_env("REMA_REWARD_QA_SIMILARITY_THRESHOLD", 0.1, min_value=0.0, max_value=1.0),
        "score_timeout_s": _int_env("REMA_REWARD_TIMEOUT_S", 3600, min_value=1),
        "show_progress_bar": _bool_env("REMA_REWARD_SHOW_TQDM", True),
    }

def parse_dia_id(dia_id):
    """Parse dia_id like 'D8:17' -> (session:int, dia:int)."""
    if not dia_id or not isinstance(dia_id, str):
        return None, None
    m = re.search(r"D(?P<sess>\d+):(?P<dia>\d+)", dia_id.strip())
    if not m:
        return None, None
    return int(m.group("sess")), int(m.group("dia"))

# Category ID to human-readable name mapping
# Used for per-category metrics (F1 and BLEU scores)
CATEGORY_NAMES = {
    1: 'multi_hop',
    2: 'temporal',
    3: 'open_domain',
    4: 'single_hop',
    5: 'adversarial',
}

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def compute_bleu(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if len(pred_tokens) == 0:
        return 0.0

    truth_count = Counter(truth_tokens)
    pred_count = Counter(pred_tokens)

    clipped = sum(min(pred_count[t], truth_count[t]) for t in pred_count)
    precision = clipped / len(pred_tokens) if pred_tokens else 0.0
    if pred_tokens and truth_tokens:
        bp = 1.0 if len(pred_tokens) >= len(truth_tokens) else math.exp(1 - len(truth_tokens)/len(pred_tokens))
    else:
        bp = 0.0
    return bp * precision

def compute_score_fn(compute_score, params):
    # data_source, response, ground_truth, extra_info = params
    qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, session_id, session_evidences, extra_info, mem_op_stats, dia_ids_affected_per_turn, cumulative_session_tokens, snapshot_suffix = params
    return compute_score(qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, session_id, session_evidences, extra_info, mem_op_stats, dia_ids_affected_per_turn, cumulative_session_tokens, snapshot_suffix)

def process_single_qa(qa_pair, memory, speakers, session_time, reward_cfg=None):
    """Process a single QA pair - to be called in parallel"""
    reward_cfg = reward_cfg or _reward_speed_config()
    question = qa_pair['question']
    gold_answer = str(qa_pair['answer']).strip()
    evidence = qa_pair.get('evidence', None)
    category = qa_pair.get('category', 0)

    # Generate prompt for memory retrieval (to get dia_ids)
    prompt, speaker_1_dia_ids, speaker_2_dia_ids = generate_qa_prompt(memory, speaker_1=speakers[0], speaker_2=speakers[1], 
                                             question=question, session_time=session_time, 
                                             top_k_per_speaker=reward_cfg.get('qa_top_k_per_speaker', 20),
                                             similarity_threshold=reward_cfg.get('qa_similarity_threshold', 0.0),
                                             use_similarity=True)
    response = judge_with_llm(prompt)
    predicted_answer = extract_answer_from_text(response)

    # Compute scores
    question_score = compute_f1(predicted_answer, gold_answer)
    bleu_score = compute_bleu(predicted_answer, gold_answer)
    
    return {
        'question': question,
        'gold_answer': gold_answer,
        'predicted_answer': predicted_answer,
        'response': response,
        'question_score': question_score,
        'bleu_score': bleu_score,
        'category': category,
        'evidence': evidence,
        'speaker_1_dia_ids': speaker_1_dia_ids,
        'speaker_2_dia_ids': speaker_2_dia_ids,
        'prompt': prompt
    }

def locomo_score(qa_pairs: list[dict], conv_id: int, chunk_id: int, speakers: list[str], epoch: int, split: str, index: int, session_time: str, session_id: int, session_evidences: list, extra_info: dict=None, mem_op_stats: dict=None, dia_ids_affected_per_turn: list=None, cumulative_session_tokens: int=None, snapshot_suffix: str = "", reward_cfg: dict=None) -> tuple[float, dict]:
    reward_cfg = reward_cfg or _reward_speed_config()
    key = f"{conv_id}_chunk{chunk_id}_epoch{epoch}"
    memory = MemoryManager().get_snapshot(
        sample_id=conv_id,
        chunk_id=chunk_id,
        epoch=epoch,
        split=split,
        index_in_batch=index,
        snapshot_suffix=snapshot_suffix,
    )
    if memory is None:
        raise RuntimeError(
            f"Missing memory snapshot for reward scoring: sample_id={conv_id}, chunk_id={chunk_id}, "
            f"epoch={epoch}, split={split}, index_in_batch={index}, snapshot_suffix={snapshot_suffix!r}"
        )
    compression_ratio = 0.0
    
    # Track memory-related metrics
    tracking_metrics = {
        'memory_size': 0,
        'memory_insert_count': 0,
        'memory_delete_count': 0,
        'memory_update_count': 0,
        'memory_operation_count': 0,
        'evidence_precision': 0.0,
        'evidence_recall': 0.0,
        'avg_retrieval_rank': 0.0,
        'memory_failure_rate': 0.0,
        'retrieval_failure_rate': 0.0,
        'total_failure_rate': 0.0,
        'memory_token_count': 0,
        'memory_compression_ratio': 0.0,
    }
    
    # Memory size: number of memory items stored
    if memory is not None and hasattr(memory, 'memories'):
        tracking_metrics['memory_size'] = len(memory.memories)
        tracking_metrics['memory_token_count'] = memory.total_tokens
        # Only penalize memory growth above a free-zone threshold (default 70% of session tokens).
        # Below the threshold the model can store facts freely with no penalty.
        # Above the threshold, penalty grows linearly with memory size.
        # Old formula (1 - mem/cumulative) was backwards: it gave ZERO penalty when
        # mem > cumulative (the worst bloated case), because max(0,...) clamped negatives.
        # New formula: (mem - threshold) / cumulative grows monotonically as memory bloats.
        compression_threshold_frac = _float_env("REMA_REWARD_COMPRESSION_THRESHOLD_FRAC", 0.7, min_value=0.0, max_value=1.0)
        compression_threshold = compression_threshold_frac * cumulative_session_tokens
        mem_tokens = memory.total_tokens
        if mem_tokens <= compression_threshold:
            compression_ratio = 0.0
        else:
            compression_ratio = (mem_tokens - compression_threshold) / cumulative_session_tokens
        tracking_metrics['memory_compression_ratio'] = compression_ratio
        print(f"[LocomoScore] Memory size for conv {conv_id}, chunk {chunk_id} snapshot_suffix: {snapshot_suffix}: {tracking_metrics['memory_size']} memory items, {tracking_metrics['memory_token_count']} tokens")
    
    # Memory operation counts (individual and total)
    if mem_op_stats is not None:
        tracking_metrics['memory_insert_count'] = mem_op_stats.get('insert_successful', 0)
        tracking_metrics['memory_delete_count'] = mem_op_stats.get('delete_successful', 0)
        tracking_metrics['memory_update_count'] = mem_op_stats.get('update_successful', 0)
        tracking_metrics['memory_operation_count'] = (
            tracking_metrics['memory_insert_count'] + 
            tracking_metrics['memory_delete_count'] + 
            tracking_metrics['memory_update_count']
        )
        # print(f"[LocomoScore] Memory operations for conv {conv_id}, chunk {chunk_id} snapshot_suffix: {snapshot_suffix}:")
        # print(f"  - Insert: {tracking_metrics['memory_insert_count']}")
        # print(f"  - Delete: {tracking_metrics['memory_delete_count']}")
        # print(f"  - Update: {tracking_metrics['memory_update_count']}")
        # print(f"  - Total operations: {tracking_metrics['memory_operation_count']}")
    
    # Optionally subsample QAs for faster experimentation.
    max_qa_per_sample = reward_cfg.get('max_qa_per_sample', 0)
    if max_qa_per_sample > 0 and len(qa_pairs) > max_qa_per_sample:
        strategy = reward_cfg.get('qa_sample_strategy', 'first')
        if strategy == 'random':
            same_qas_across_rollouts = reward_cfg.get('same_qas_across_rollouts', False)
            qa_sample_seed = reward_cfg.get('qa_sample_seed', 0)
            if same_qas_across_rollouts:
                seed_tuple = (qa_sample_seed, conv_id, chunk_id, epoch, session_id)
            else:
                seed_tuple = (qa_sample_seed, conv_id, chunk_id, epoch, index, session_id)
            rng = random.Random(hash(seed_tuple) & 0xFFFFFFFF)
            qa_pairs = rng.sample(qa_pairs, max_qa_per_sample)
        else:
            qa_pairs = qa_pairs[:max_qa_per_sample]

    # Compute score for all selected QA pairs and return average
    qa_scores = 0.0
    bleu_scores = 0.0
    num_questions = len(qa_pairs)
    
    # For computing retrieval quality metrics
    total_evidence_precision = 0.0
    total_evidence_recall = 0.0
    total_avg_rank = 0.0
    total_retrieval_failures = 0
    total_memory_failures = 0  # Evidence not in memory
    total_retrieval_only_failures = 0  # Evidence in memory but not retrieved
    num_questions_with_evidence = 0

    # Per-category tracking
    category_f1_scores = {}  # {category: [f1_scores]}
    category_bleu_scores = {}  # {category: [bleu_scores]}
    
    # Per-session tracking for dense rewards
    session_f1_scores = {}  # {session_id: [f1_scores]}

    # Calculate session-level evidence coverage
    # Compare memory's dia_ids_set against session_evidences needed for this session
    if session_evidences and hasattr(memory, 'dia_ids_set'):
        session_evidences_set = set(session_evidences)
        covered_evidences = memory.dia_ids_set.intersection(session_evidences_set)
        evidence_retrieval_coverage = len(covered_evidences) / len(session_evidences_set) if len(session_evidences_set) > 0 else 0.0
        # print(f"[LocomoScore] Session {session_id} evidence coverage: {len(covered_evidences)}/{len(session_evidences_set)} ({evidence_retrieval_coverage:.3f})")
        if len(covered_evidences) < len(session_evidences_set):
            missing = session_evidences_set - covered_evidences
            # print(f"[LocomoScore] Missing evidence dia_ids: {sorted(list(missing))[:10]}...")  # Show first 10
    else:
        evidence_retrieval_coverage = 0.0
        # print(f"[LocomoScore] No session_evidences or dia_ids_set available, evidence coverage = 0.0")
    
    # print(f"[LocomoScore] Processing {num_questions} questions for conv {conv_id}, chunk {chunk_id}")
    
    # Handle different question counts
    if num_questions == 0:
        # No questions - empty results will naturally produce zero scores
        results = []
    else:
        # Use parallel processing for any number of questions
        # print(f"[LocomoScore] Using parallel processing for {num_questions} questions")
        max_inner_workers = max(1, reward_cfg.get('max_inner_workers', 4))
        worker_count = min(num_questions, max_inner_workers)
        if worker_count == 1:
            results = [process_single_qa(qa_pair, memory, speakers, session_time, reward_cfg=reward_cfg) for qa_pair in qa_pairs]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(process_single_qa, qa_pair, memory, speakers, session_time, reward_cfg)
                           for qa_pair in qa_pairs]
                results = [future.result() for future in futures]
    
    # Optional per-QA dump for post-hoc LLM-judge scoring (REMA_DUMP_QA=1).
    # Layout: $REMA_QA_DUMP_DIR/$REMA_RUN_NAME/$split/step_unknown/convconv-<c>_chunk<k>_epoch<e>_idx<i>.jsonl
    # Schema matches scripts/.../score_locomo_qa_dumps.py expectations.
    _dump_qa = os.environ.get('REMA_DUMP_QA', '').strip()
    if _dump_qa and _dump_qa not in ('0', 'false', 'False', ''):
        _dump_splits = [s.strip() for s in os.environ.get('REMA_QA_DUMP_SPLITS', 'test,val').split(',') if s.strip()]
        if split in _dump_splits and len(results) > 0:
            _dump_dir = os.environ.get('REMA_QA_DUMP_DIR', '').strip() or os.path.join(os.getcwd(), 'qa_dumps')
            _run_name = os.environ.get('REMA_RUN_NAME', 'unnamed_run').strip() or 'unnamed_run'
            _out_dir = os.path.join(_dump_dir, _run_name, split, 'step_unknown')
            def _to_native(x):
                # Recursively convert numpy / tensor types to JSON-serializable Python primitives.
                try:
                    import numpy as _np
                    if isinstance(x, _np.ndarray):
                        return [_to_native(v) for v in x.tolist()]
                    if isinstance(x, (_np.integer,)):
                        return int(x)
                    if isinstance(x, (_np.floating,)):
                        return float(x)
                    if isinstance(x, (_np.bool_,)):
                        return bool(x)
                except Exception:
                    pass
                if isinstance(x, (list, tuple)):
                    return [_to_native(v) for v in x]
                if isinstance(x, dict):
                    return {str(k): _to_native(v) for k, v in x.items()}
                if hasattr(x, 'tolist'):
                    try:
                        return _to_native(x.tolist())
                    except Exception:
                        return str(x)
                return x
            try:
                os.makedirs(_out_dir, exist_ok=True)
                _out_file = os.path.join(_out_dir, f"convconv-{conv_id}_chunk{chunk_id}_epoch{epoch}_idx{index}.jsonl")
                with open(_out_file, 'w') as _f:
                    for _qa_idx, _r in enumerate(results):
                        _rec = {
                            'qa_idx': _qa_idx,
                            'split': split,
                            'conv_id': f"conv-{conv_id}" if not str(conv_id).startswith('conv-') else str(conv_id),
                            'chunk_id': int(chunk_id) if chunk_id is not None else None,
                            'epoch': int(epoch) if epoch is not None else None,
                            'index_in_batch': int(index) if index is not None else None,
                            'session_id': int(session_id) if session_id is not None else None,
                            'session_time': str(session_time) if session_time is not None else '',
                            'speakers': [str(s) for s in (list(speakers) if speakers is not None else [])],
                            'question': str(_r.get('question', '')),
                            'gold_answer': str(_r.get('gold_answer', '')),
                            'predicted_answer': str(_r.get('predicted_answer', '')),
                            'response': str(_r.get('response', '')),
                            'category': int(_r.get('category', 0)) if _r.get('category') is not None else 0,
                            'evidence': _to_native(_r.get('evidence')),
                            'f1': float(_r.get('question_score', 0.0)),
                            'bleu': float(_r.get('bleu_score', 0.0)),
                        }
                        _f.write(json.dumps(_rec, ensure_ascii=False) + '\n')
            except Exception as _e:
                print(f"[LocomoScore] QA dump failed for conv={conv_id} chunk={chunk_id}: {_e}")

    # Process results
    for qa_idx, result in enumerate(results):
        question_score = result['question_score']
        bleu_score = result['bleu_score']
        category = result['category']
        # dia_ids retrieved from memory retrieval part (separate for each speaker), and evidence needed to solve Q
        speaker_1_dia_ids = result['speaker_1_dia_ids']
        speaker_2_dia_ids = result['speaker_2_dia_ids']
        dia_ids_needed_for_q = result['evidence']
        
        # Track per-category scores for aggregation later
        if category not in category_f1_scores:
            category_f1_scores[category] = []
            category_bleu_scores[category] = []
        category_f1_scores[category].append(question_score)
        category_bleu_scores[category].append(bleu_score)

        # Track per-session scores for dense rewards
        # Extract sessions from dia_ids_needed_for_q
        q_sessions = set()
        for ev_id in (dia_ids_needed_for_q or []):
            s, _ = parse_dia_id(ev_id)
            if s is not None:
                q_sessions.add(s)
        
        # If no evidence session found, default to current session
        if not q_sessions:
            q_sessions.add(session_id)
            
        for s in q_sessions:
            if s not in session_f1_scores:
                session_f1_scores[s] = []
            session_f1_scores[s].append(question_score)

        # Accumulate scores for averaging
        qa_scores += question_score
        bleu_scores += bleu_score
        
        # print(f"[LocomoScore] Q{qa_idx+1}/{num_questions} [{CATEGORY_NAMES[category]}]: {result['question']}")
        # print(f"[LocomoScore] Gold: {result['gold_answer']}, Predicted: {result['predicted_answer']}, F1: {question_score}, BLEU: {bleu_score}")
        # print(f"[LocomoScore] Speaker 1 dia_ids (ranked): {speaker_1_dia_ids}")
        # print(f"[LocomoScore] Speaker 2 dia_ids (ranked): {speaker_2_dia_ids}")

        # Track retrieval quality metrics for each question
        if dia_ids_needed_for_q and len(dia_ids_needed_for_q) > 0:
            dia_ids_needed_for_q = dia_ids_needed_for_q or []
            speaker_1_dia_ids = speaker_1_dia_ids or []
            speaker_2_dia_ids = speaker_2_dia_ids or []
            dia_ids_retrieved_combined = speaker_1_dia_ids + speaker_2_dia_ids
            
            needed_set = set(dia_ids_needed_for_q)
            retrieved_set = set(dia_ids_retrieved_combined)
            correctly_retrieved = needed_set & retrieved_set
            
            # Evidence precision: % of retrieved that are relevant
            if len(retrieved_set) > 0:
                precision = len(correctly_retrieved) / len(retrieved_set)
                total_evidence_precision += precision
            
            # Evidence recall: % of needed that are retrieved
            if len(needed_set) > 0:
                recall = len(correctly_retrieved) / len(needed_set)
                total_evidence_recall += recall
            
            # Track root causes of failures
            memory_set = memory.dia_ids_set if hasattr(memory, 'dia_ids_set') else set()
            not_in_memory = needed_set - memory_set  # Memory problem
            in_memory_but_not_retrieved = (needed_set & memory_set) - retrieved_set  # Retrieval problem
            
            total_memory_failures += len(not_in_memory)
            total_retrieval_only_failures += len(in_memory_but_not_retrieved)
            total_retrieval_failures += len(needed_set - retrieved_set)  # Total (both problems)
            
            # Average rank of needed evidence in retrieval results
            ranks = []
            for needed_id in needed_set:
                if needed_id in dia_ids_retrieved_combined:
                    rank = dia_ids_retrieved_combined.index(needed_id) + 1
                    ranks.append(rank)
            
            if len(ranks) > 0:
                total_avg_rank += sum(ranks) / len(ranks)
            
            num_questions_with_evidence += 1
        
        if question_score < 1.0:
            # print(f"[LocomoScore] === Mismatch Analysis (F1={question_score:.3f}) ===")
            
            # Handle None/empty cases
            dia_ids_needed_for_q = dia_ids_needed_for_q or []
            speaker_1_dia_ids = speaker_1_dia_ids or []
            speaker_2_dia_ids = speaker_2_dia_ids or []
            
            # Combine both speakers for overall analysis
            dia_ids_retrieved_combined = speaker_1_dia_ids + speaker_2_dia_ids
            
            # Convert to sets for analysis
            needed_set = set(dia_ids_needed_for_q)
            retrieved_set = set(dia_ids_retrieved_combined)
            memory_set = memory.dia_ids_set if hasattr(memory, 'dia_ids_set') else set()
            
            # print(f"[LocomoScore] Evidence needed: {sorted(list(needed_set))} (total: {len(needed_set)})")
            # print(f"[LocomoScore] Speaker 1 retrieved: {speaker_1_dia_ids} (total: {len(speaker_1_dia_ids)})")
            # print(f"[LocomoScore] Speaker 2 retrieved: {speaker_2_dia_ids} (total: {len(speaker_2_dia_ids)})")
            # print(f"[LocomoScore] Evidence in memory: (total: {len(memory_set)})")
            
            # Compute diagnostic metrics
            correctly_retrieved = needed_set & retrieved_set  # Needed AND retrieved
            missing_from_retrieval = needed_set - retrieved_set  # Needed but NOT retrieved
            extra_retrieved = retrieved_set - needed_set  # Retrieved but NOT needed
            in_memory_set = needed_set & memory_set
            not_in_memory_set = needed_set - memory_set
            retrieval_failure_set = in_memory_set - retrieved_set
            
            # Report coverage metrics
            if len(needed_set) > 0:
                coverage = len(correctly_retrieved) / len(needed_set)
                precision = len(correctly_retrieved) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
                # print(f"[LocomoScore] Evidence coverage (recall): {coverage:.1%} ({len(correctly_retrieved)}/{len(needed_set)} needed retrieved)")
                # print(f"[LocomoScore] Evidence precision: {precision:.1%} ({len(correctly_retrieved)}/{len(retrieved_set)} retrieved were relevant)")
                # if len(extra_retrieved) > 0:
                    # print(f"[LocomoScore] Extra (irrelevant) retrievals: {len(extra_retrieved)} items not needed")
                    # pass
            
            # Report issues
            # if len(not_in_memory_set) > 0:
                # print(f"[LocomoScore] MEMORY PROBLEM: {len(not_in_memory_set)}/{len(needed_set)} needed dia_ids not saved in memory")
                # print(f"[LocomoScore] Missing from memory: {sorted(list(not_in_memory_set))}")
                # pass
            
            # if len(retrieval_failure_set) > 0:
                # print(f"[LocomoScore] RETRIEVAL PROBLEM: {len(retrieval_failure_set)}/{len(in_memory_set)} dia_ids in memory but not retrieved")
                # print(f"[LocomoScore] In memory but not retrieved: {sorted(list(retrieval_failure_set))}")
                # pass
            
            # Show ranking analysis
            if len(needed_set) > 0:
                # print(f"[LocomoScore] === Ranking Analysis (F1={question_score:.3f}) ===")
                for needed_id in sorted(list(needed_set)):
                    if needed_id in speaker_1_dia_ids:
                        rank = speaker_1_dia_ids.index(needed_id) + 1
                        # print(f"[LocomoScore]   dia_id '{needed_id}' found in Speaker 1 at rank {rank}/{len(speaker_1_dia_ids)}")
                        pass
                    elif needed_id in speaker_2_dia_ids:
                        rank = speaker_2_dia_ids.index(needed_id) + 1
                        # print(f"[LocomoScore]   dia_id '{needed_id}' found in Speaker 2 at rank {rank}/{len(speaker_2_dia_ids)}")
                        pass
                    else:
                        # print(f"[LocomoScore]   dia_id '{needed_id}' NOT RETRIEVED from either speaker")
                        pass
            
            # Show retrieved memory context
            # if 'prompt' in result and result['prompt']:
            #     retrieved_memory_idx = result['prompt'].find("Memories for user")
            #     if retrieved_memory_idx != -1:
            #         memory_section = result['prompt'][retrieved_memory_idx:]
            #         print(f"[LocomoScore] Retrieved memory preview:\n{memory_section}...\n")
    
    # Calculate average scores
    avg_f1_score = qa_scores / num_questions if num_questions > 0 else 0.0
    avg_bleu_score = bleu_scores / num_questions if num_questions > 0 else 0.0
    # print(f"[LocomoScore] Average F1 score: {avg_f1_score:.3f} ({qa_scores}/{num_questions})")
    # print(f"[LocomoScore] Average BLEU score: {avg_bleu_score:.3f} ({bleu_scores}/{num_questions})")
    # print(f"[LocomoScore] Session evidence coverage: {evidence_retrieval_coverage:.3f}")
    
    # Compute average retrieval quality metrics
    if num_questions_with_evidence > 0:
        tracking_metrics['evidence_precision'] = total_evidence_precision / num_questions_with_evidence
        tracking_metrics['evidence_recall'] = total_evidence_recall / num_questions_with_evidence
        tracking_metrics['avg_retrieval_rank'] = total_avg_rank / num_questions_with_evidence

        # Compute failure rates by root cause
        total_needed_evidence = sum(len(qa_pair.get('evidence', [])) for qa_pair in qa_pairs if qa_pair.get('evidence'))
        if total_needed_evidence > 0:
            # Memory problem: evidence not saved in memory
            tracking_metrics['memory_failure_rate'] = total_memory_failures / total_needed_evidence
            # Retrieval problem: evidence in memory but not retrieved
            tracking_metrics['retrieval_failure_rate'] = total_retrieval_only_failures / total_needed_evidence
            # Total failure: combines both problems (memory + retrieval)
            tracking_metrics['total_failure_rate'] = total_retrieval_failures / total_needed_evidence

        # Detailed per-chunk metrics line — parsed by autonomous loop / scripts to extract MemTok/Comp/MFail aggregates.
        try:
            print(f"[LocomoScore] conv_id={conv_id} chunk_id={chunk_id} mem_size={tracking_metrics.get('memory_size', 0)} mem_tokens={tracking_metrics.get('memory_token_count', 0)} comp_ratio={tracking_metrics.get('memory_compression_ratio', 0.0):.4f} mfail={tracking_metrics.get('memory_failure_rate', 0.0):.4f} retrieval_fail={tracking_metrics.get('retrieval_failure_rate', 0.0):.4f} total_fail={tracking_metrics.get('total_failure_rate', 0.0):.4f} ev_prec={tracking_metrics.get('evidence_precision', 0.0):.4f} ev_recall={tracking_metrics.get('evidence_recall', 0.0):.4f}")
        except Exception:
            pass
        
        # print(f"[LocomoScore] Retrieval quality metrics:")
        # print(f"  - Evidence precision: {tracking_metrics['evidence_precision']:.3f}")
        # print(f"  - Evidence recall: {tracking_metrics['evidence_recall']:.3f}")
        # print(f"  - Avg retrieval rank: {tracking_metrics['avg_retrieval_rank']:.1f}")
        # print(f"[LocomoScore] Failure analysis (root causes):")
        # print(f"  - Memory failure rate: {tracking_metrics.get('memory_failure_rate', 0.0):.3f} ({total_memory_failures}/{total_needed_evidence} evidence not in memory)")
        # print(f"  - Retrieval failure rate: {tracking_metrics.get('retrieval_failure_rate', 0.0):.3f} ({total_retrieval_only_failures}/{total_needed_evidence} in memory but not retrieved)")
        # print(f"  - Total failure rate: {tracking_metrics.get('total_failure_rate', 0.0):.3f} ({total_retrieval_failures}/{total_needed_evidence} total failures)")

    # Return raw category scores (not averages) for global aggregation
    category_raw_scores = {
        'f1_scores': category_f1_scores,  # {category: [individual_f1_scores]}
        'bleu_scores': category_bleu_scores  # {category: [individual_bleu_scores]}
    }
    
    # Compute turn-level rewards based on dia_id causality
    turn_level_f1_rewards = None
    turn_level_bleu_rewards = None
    
    if dia_ids_affected_per_turn is not None and len(dia_ids_affected_per_turn) > 0:
        # print(f"[LocomoScore] Computing turn-level rewards from dia_id causality...")
        # print(f"[LocomoScore] dia_ids_affected_per_turn: {dia_ids_affected_per_turn}")
        
        # Build mapping: dia_id -> turn_id (which turn affected this dia_id)
        dia_id_to_turn = {}
        max_turn_id = 0
        for turn_info in dia_ids_affected_per_turn:
            turn_id = turn_info['turn_id']
            dia_ids = turn_info['dia_ids']
            max_turn_id = max(max_turn_id, turn_id)
            for dia_id in dia_ids:
                if dia_id not in dia_id_to_turn:
                    dia_id_to_turn[dia_id] = []
                dia_id_to_turn[dia_id].append(turn_id)
        
        # print(f"[LocomoScore] Built dia_id_to_turn mapping with {len(dia_id_to_turn)} dia_ids across {max_turn_id+1} turns")
        
        # Initialize turn-level reward accumulators
        turn_f1_scores = {turn_id: [] for turn_id in range(max_turn_id + 1)}
        turn_bleu_scores = {turn_id: [] for turn_id in range(max_turn_id + 1)}
        
        # Assign QA scores to turns based on evidence requirements
        # TODO:: should we do it this way based on what our memory did in turn i, or assign directly to the turn
        # that was responsible for that evidence?? even though it worked on it or not ? 
        for qa_idx, result in enumerate(results):
            question_score = result['question_score']
            bleu_score = result['bleu_score']
            dia_ids_needed = result['evidence']
            
            if not dia_ids_needed or len(dia_ids_needed) == 0:
                continue
            
            # Find which turns affected the dia_ids needed for this question
            relevant_turns = set()
            for dia_id in dia_ids_needed:
                if dia_id in dia_id_to_turn:
                    relevant_turns.update(dia_id_to_turn[dia_id])
            
            if len(relevant_turns) > 0:
                # Case: Turns created the needed dia_ids
                # Assign this QA's scores to all relevant turns
                for turn_id in relevant_turns:
                    turn_f1_scores[turn_id].append(question_score)
                    turn_bleu_scores[turn_id].append(bleu_score)
                # print(f"[LocomoScore] Q{qa_idx+1} (F1={question_score:.3f}) needs dia_ids {dia_ids_needed} → assigned to turns {sorted(relevant_turns)}")
            else:
                # VERY IMPORTANT CASE: Evidence needed but no turn created it --> penalize on trajectory level (last turn)
                # Case: Evidence needed but no turn created it - assign to last turn as trajectory-level feedback
                # This will be back-propagated through discounted returns
                turn_f1_scores[max_turn_id].append(question_score)
                turn_bleu_scores[max_turn_id].append(bleu_score)
                # print(f"[LocomoScore] Q{qa_idx+1} (F1={question_score:.3f}) needs dia_ids {dia_ids_needed} but no turn created them → assigned to last turn {max_turn_id} as trajectory signal")
        
        # Compute average scores per turn
        turn_level_f1_rewards = []
        turn_level_bleu_rewards = []
        for turn_id in range(max_turn_id + 1):
            if len(turn_f1_scores[turn_id]) > 0:
                avg_f1 = sum(turn_f1_scores[turn_id]) / len(turn_f1_scores[turn_id])
                avg_bleu = sum(turn_bleu_scores[turn_id]) / len(turn_bleu_scores[turn_id])
                turn_level_f1_rewards.append(avg_f1)
                turn_level_bleu_rewards.append(avg_bleu)
                # print(f"[LocomoScore] Turn {turn_id}: F1={avg_f1:.3f} (avg of {len(turn_f1_scores[turn_id])} QAs), BLEU={avg_bleu:.3f}")
            else:
                # No QA pairs mapped to this turn - give 0 reward
                turn_level_f1_rewards.append(0.0)
                turn_level_bleu_rewards.append(0.0)
                # print(f"[LocomoScore] Turn {turn_id}: No QA pairs mapped → reward=0.0")
        
        # Apply insertion penalty to last turn reward (penalizes wasteful insertions)
        # This encourages efficient memory usage without making reward non-stationary
        # if turn_level_f1_rewards and len(turn_level_f1_rewards) > 0:
        #     num_insertions = mem_op_stats.get('insert_successful', 0) if mem_op_stats else 0
        #     lambda_insertion = 0.01  # Penalty per insertion
        #     insertion_penalty = lambda_insertion * num_insertions
        #     turn_level_f1_rewards[-1] = max(0.0, turn_level_f1_rewards[-1] - insertion_penalty)
        #     turn_level_bleu_rewards[-1] = max(0.0, turn_level_bleu_rewards[-1] - insertion_penalty)
        #     print(f"[LocomoScore] === Insertion Penalty (Applied to Last Turn) ===")
        #     print(f"[LocomoScore] Successful insertions in trajectory: {num_insertions}")
        #     print(f"[LocomoScore] Penalty: λ * insertions = {lambda_insertion} * {num_insertions} = {insertion_penalty:.4f}")
        #     print(f"[LocomoScore] Last turn F1 after penalty: {turn_level_f1_rewards[-1]:.3f}")
    else:
        # print(f"[LocomoScore] No dia_ids_affected_per_turn provided, skipping turn-level reward computation")
        pass

    memory_info = {
        "key": key,
        "memory": memory,
        "conv_id": conv_id,
        "chunk_id": chunk_id,
        "epoch": epoch,
        "split": split,
        "snapshot_suffix": snapshot_suffix,
    }
    
    # Compute average scores per session
    per_session_f1 = {}
    for s, scores in session_f1_scores.items():
        per_session_f1[s] = sum(scores) / len(scores) if scores else 0.0

    # Cache is never used, since we always get new stuff due to (generation, retrieval, ... ) randomness
    # Force merge cache before returning (critical for ProcessPool workers)
    # from verl.rema_trainer.memory.judge_llm import merge_to_main_cache
    # merge_to_main_cache()
    
    # Return F1 score, BLEU score, evidence score, category_raw_scores, memory_info, tracking_metrics, turn_level_f1_rewards, turn_level_bleu_rewards and per_session_f1
    return avg_f1_score, compression_ratio, avg_bleu_score, evidence_retrieval_coverage, category_raw_scores, memory_info, tracking_metrics, turn_level_f1_rewards, turn_level_bleu_rewards, per_session_f1

class ReMARewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, top_k_percentage=0.5) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score or _default_compute_score
        self.reward_speed_cfg = _reward_speed_config()
        self.compute_score = partial(locomo_score, reward_cfg=self.reward_speed_cfg)
        self.top_k_percentage = top_k_percentage  # sample from top k% of memories (e.g., 0.3 = top 30%)

    def _phase_reward_cfg(self, data: DataProto) -> dict:
        cfg = dict(self.reward_speed_cfg)
        split = str(data.meta_info.get('split', 'train')).lower()

        # Inner batches are explicitly tagged with uid prefix "inner_" in trainer code.
        is_inner_batch = False
        if split == 'train' and 'uid' in data.non_tensor_batch:
            try:
                uids = data.non_tensor_batch['uid']
                if len(uids) > 0:
                    is_inner_batch = str(uids[0]).startswith('inner_')
            except Exception:
                is_inner_batch = False

        override = -1
        if split in {'validation', 'test'}:
            override = cfg.get('max_qa_eval', -1)
        elif split == 'train' and is_inner_batch:
            override = cfg.get('max_qa_train_inner', -1)
        elif split == 'train':
            override = cfg.get('max_qa_train_terminal', -1)

        if override is not None and override >= 0:
            cfg['max_qa_per_sample'] = override
        else:
            cfg.pop('max_qa_per_sample', None)
        return cfg

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, compression_penalty: float)-> Dict[str, torch.Tensor]:
        """We will expand this function gradually based on the available datasets"""

        # print("\n" + "="*80)
        # print("REWARD MANAGER __call__ STARTED")
        # print("="*80)
        # print(f"[RewardManager] Input data batch size: {len(data)}")
        # print(f"[RewardManager] data.batch keys: {list(data.batch.keys())}")
        # print(f"[RewardManager] data.non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")
        # print(f"[RewardManager] data.meta_info keys: {list(data.meta_info.keys())}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            # print("[RewardManager] Found pre-computed rm_scores, returning directly")
            return data.batch['rm_scores']
        
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']
        # print(f"[RewardManager] batch_size: {batch_size}, max_num_turns: {max_num_turns}")

        
        agent_roles = data.meta_info['agent_roles']
        # print(f"[RewardManager] agent_roles: {agent_roles}")
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) for role in agent_roles
        }
        # print(f"[RewardManager] Initialized reward_tensor_map with keys: {list(reward_tensor_map.keys())}")
        # for key, tensor in reward_tensor_map.items():
            # print(f"[RewardManager] {key} shape: {tensor.shape}")
        
        already_print_data_sources = {}
        memory_manager = MemoryManager()

        phase_cfg = self._phase_reward_cfg(data)
        phase_compute_score = partial(locomo_score, reward_cfg=phase_cfg)
        
        # print(f"\n[RewardManager] Preparing parameters for score computation...")
        params = [
            (
             json.loads(data[i].non_tensor_batch['qa_pairs_json']),
             data[i].non_tensor_batch['sample_id'],
             data[i].non_tensor_batch['chunk_id'],
             data[i].non_tensor_batch['speakers'],
             data[i].batch['epoch'],
             data.meta_info['split'],
             data[i].batch['rollout_idx'],  # Use rollout_idx computed AFTER repeating
             data[i].non_tensor_batch['session_time'],
             data[i].non_tensor_batch['session_id'],  # Session ID for evidence tracking
             json.loads(data[i].non_tensor_batch.get('session_evidences_json', '[]')),  # Session evidences needed
             data[i].non_tensor_batch.get('extra_info', None),
             # Memory operation statistics
             {
                 'insert_successful': data[i].non_tensor_batch.get('mem_insert_successful', 0),
                 'delete_successful': data[i].non_tensor_batch.get('mem_delete_successful', 0),
                 'update_successful': data[i].non_tensor_batch.get('mem_update_successful', 0),
             },
             # dia_ids affected per turn for causal reward assignment
             # Convert numpy array back to list if needed
             (list(data[i].non_tensor_batch.get('dia_ids_affected_per_turn', [])) 
              if data[i].non_tensor_batch.get('dia_ids_affected_per_turn') is not None 
              else None),
             data[i].non_tensor_batch.get('cumulative_session_tokens', 0),
             data.meta_info.get('memory_snapshot_suffix', ''),
             )
            for i in range(len(data))
        ]

        qa_scores = []  # Track QA accuracy (F1) separately
        compression_ratios = []  # Track compression ratios separately
        bleu_scores = []  # Track BLEU scores separately
        evidence_scores = [] # Track evidence scores separately
        category_stats_list = []  # Track per-category stats
        memory_infos = []  # Collect memory info from each result
        tracking_metrics_list = []  # Track additional metrics (memory size, retrieval quality, etc.)
        turn_level_f1_list = []  # Track turn-level F1 rewards
        turn_level_bleu_list = []  # Track turn-level BLEU rewards
        per_session_f1_list = []  # Track per-session F1 rewards
        # print(f"\n[RewardManager] Starting score computation with ThreadPool...")
        max_outer_workers = self.reward_speed_cfg.get('max_outer_workers', 16)
        score_timeout_s = self.reward_speed_cfg.get('score_timeout_s', 3600)
        show_progress_bar = self.reward_speed_cfg.get('show_progress_bar', True)
        with ThreadPool(max_workers=max_outer_workers) as pool:
            future = pool.map(partial(compute_score_fn, phase_compute_score), params, timeout=score_timeout_s)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores", disable=not show_progress_bar) as pbar:
                while True:
                    try:
                        result = next(iterator)
                        # Updated format: (f1_score, bleu_score, evidence_score, category_stats, memory_info, tracking_metrics, turn_level_f1, turn_level_bleu, per_session_f1)
                        qa_score, comp_ratio, bleu_score, evidence_score, category_stats, memory_info, tracking_metrics, turn_f1, turn_bleu, per_session_f1 = result
                        qa_scores.append(qa_score)
                        compression_ratios.append(comp_ratio)
                        bleu_scores.append(bleu_score)
                        evidence_scores.append(evidence_score)
                        category_stats_list.append(category_stats)
                        memory_infos.append(memory_info)
                        tracking_metrics_list.append(tracking_metrics)
                        turn_level_f1_list.append(turn_f1)
                        turn_level_bleu_list.append(turn_bleu)
                        per_session_f1_list.append(per_session_f1)
                    except TimeoutError:
                        # print('[RewardManager] Time Out')
                        qa_scores.append(0.0)
                        compression_ratios.append(0.0)
                        bleu_scores.append(0.0)
                        evidence_scores.append(0.0)
                        category_stats_list.append({'f1_scores': {}, 'bleu_scores': {}})
                        memory_infos.append(None)
                        tracking_metrics_list.append({})
                        turn_level_f1_list.append(None)
                        turn_level_bleu_list.append(None)
                        per_session_f1_list.append({})
                    except TimeoutException:
                        # print('[RewardManager] Math verify internal timeout')
                        qa_scores.append(0.0)
                        compression_ratios.append(0.0)
                        bleu_scores.append(0.0)
                        evidence_scores.append(0.0)
                        category_stats_list.append({'f1_scores': {}, 'bleu_scores': {}})
                        memory_infos.append(None)
                        tracking_metrics_list.append({})
                        turn_level_f1_list.append(None)
                        turn_level_bleu_list.append(None)
                        per_session_f1_list.append({})
                    except StopIteration:
                        break
                    except Exception as e:
                        # print(f"[RewardManager] Error: {e}")
                        raise e
                    pbar.update(1)
        # print(f"[RewardManager] Score computation complete. Got {len(qa_scores)} F1 scores, {len(bleu_scores)} BLEU scores, and {len(evidence_scores)} evidence scores")
        
        # Compute combined scores once (single place for reward logic)
        # Give 0 score to incomplete trajectories (turn_finished != 1)
        combined_scores = []
        num_incomplete = 0
        for i in range(len(qa_scores)):
            turn_finished = data[i].batch[f'{agent_roles[0]}_turn_finished'].item()

            # Only apply masking if enabled
            if data[i].meta_info['mask_unfinished_reward']:
                if turn_finished == 1 or max_num_turns == 1:  # Successfully completed
                    # Combined reward = QA quality minus memory-bloat penalty.
                    score = qa_scores[i] - compression_penalty * compression_ratios[i]
                else:  # Incomplete trajectory - give 0 score
                    score = 0.0
                    num_incomplete += 1
            else:
                # No masking - use score regardless of completion status
                score = qa_scores[i] - compression_penalty * compression_ratios[i]
            combined_scores.append(score)
        # print(f"[RewardManager] Combined scores computed: mean={sum(combined_scores)/len(combined_scores):.4f}")
        # print(f"[RewardManager] Incomplete trajectories (turn_finished != 1): {num_incomplete}/{len(combined_scores)}")
        
        # Build memory_score_dict from collected results using combined scores
        memory_score_dict = {}
        for i, memory_info in enumerate(memory_infos):
            if memory_info is not None and memory_info["memory"] is not None:
                key = memory_info["key"]
                if key not in memory_score_dict:
                    memory_score_dict[key] = []
                memory_score_dict[key].append((combined_scores[i], memory_info))
        # print(f"[RewardManager] Built memory_score_dict with {len(memory_score_dict)} unique conversation-chunk pairs")
        
        assert len(evidence_scores) == len(data)
        assert len(qa_scores) == len(data)
        assert len(compression_ratios) == len(data)
        assert len(bleu_scores) == len(data)
        assert len(category_stats_list) == len(data)
        accuracy = torch.tensor(qa_scores, dtype=torch.float32) # bsz - F1 accuracy
        bleu = torch.tensor(bleu_scores, dtype=torch.float32) # bsz - BLEU scores
        evidence = torch.tensor(evidence_scores, dtype=torch.float32) # bsz - evidence coverage scores
        reward_tensor_map['acc'] = accuracy
        reward_tensor_map['bleu'] = bleu
        reward_tensor_map['evidence'] = evidence
        
        # Construct per-session F1 reward tensor
        # max_sessions is passed via session_id of current batch (which is the last session)
        # Find the max session_id across ALL items in the batch to avoid truncation/crashes
        all_session_ids = []
        for d in data:
            s_id = d.non_tensor_batch['session_id']
            if isinstance(s_id, torch.Tensor):
                s_id = s_id.item()
            all_session_ids.append(s_id)
        
        max_sessions = max(all_session_ids) if all_session_ids else 0
        per_session_f1_tensor = torch.zeros(batch_size, max_sessions)
        for i, ps_f1 in enumerate(per_session_f1_list):
            for s_id, score in ps_f1.items():
                if 1 <= s_id <= max_sessions:
                    per_session_f1_tensor[i, s_id - 1] = score
        
        # Apply compression penalty to per-session F1 (use final compression ratio for all sessions)
        # This ensures per-session and cumulative rewards also incentivize memory efficiency
        for i in range(batch_size):
            # Keep per-session rewards session-specific; operation shaping is applied during
            # trajectory propagation using each session's own mem_* stats.
            per_session_f1_tensor[i] = per_session_f1_tensor[i] - compression_penalty * compression_ratios[i]
        
        reward_tensor_map['per_session_f1'] = per_session_f1_tensor

        # Now saving the cumulative per session reward like session i affects all future sessions i+1,i+2,...
        # reward of session i = mean(r_i, r_{i+1}, ..., r_n)
        # We need a suffix sum (sum from i to end), but cumsum gives a prefix sum (sum from start to i).
        # Trick: Flip the tensor -> cumsum -> Flip back
        # [r1, r2, r3] -> flip -> [r3, r2, r1] -> cumsum -> [r3, r3+r2, r3+r2+r1] -> flip -> [r1+r2+r3, r2+r3, r3]
        num_sessions = per_session_f1_tensor.shape[1]
        cumulative_per_session_f1_tensor = torch.flip(torch.cumsum(torch.flip(per_session_f1_tensor, dims=[1]), dim=1), dims=[1])
        # Normalize by the number of sessions summed: session i sums (N - i) elements
        # This keeps rewards in [0, 1] range (average future F1 instead of sum)
        num_remaining = torch.arange(num_sessions, 0, -1, dtype=torch.float32).unsqueeze(0)  # [1, N] -> [N, N-1, ..., 1]
        cumulative_per_session_f1_tensor = cumulative_per_session_f1_tensor / num_remaining
        reward_tensor_map['cumulative_per_session_f1'] = cumulative_per_session_f1_tensor

        # Populate turn-level rewards from dia_id-based causality
        # print(f"\n[RewardManager] Populating turn-level rewards from dia_id causality...")
        # for i in range(batch_size):
        #     turn_f1 = turn_level_f1_list[i]
        #     turn_bleu = turn_level_bleu_list[i]
        #     turn_finished = data[i].batch[f'{agent_roles[0]}_turn_finished'].item()
            
        #     # Apply masking: if trajectory incomplete AND masking enabled, zero out turn rewards
        #     should_mask = (turn_finished != 1) and data[i].meta_info['mask_unfinished_reward']
            
        #     if turn_f1 is not None and len(turn_f1) > 0:
        #         num_turns_computed = len(turn_f1)
        #         # print(f"[RewardManager] Sample {i}: Got {num_turns_computed} turn-level F1 rewards: {[f'{x:.3f}' for x in turn_f1]}")
                
        #         # Assign to all roles (both agents benefit from good memory operations)
        #         for role in agent_roles:
        #             # Use turn-level F1 rewards (can also combine with BLEU if desired)
        #             for turn_idx in range(min(num_turns_computed, max_num_turns)):
        #                 if should_mask:
        #                     # Incomplete trajectory: zero out turn rewards (consistent with combined_scores masking)
        #                     reward_tensor_map[f'{role}_turn_level_reward'][i, turn_idx] = 0.0
        #                 else:
        #                     # Complete trajectory: use computed turn rewards
        #                     reward_tensor_map[f'{role}_turn_level_reward'][i, turn_idx] = turn_f1[turn_idx]

        #         if should_mask:
        #             pass
        #             # print(f"[RewardManager] Sample {i}: Masked turn-level rewards (turn_finished={turn_finished}, masking enabled)")
        #     else:
        #         # print(f"[RewardManager] Sample {i}: No turn-level rewards computed (using fallback)")
        #         # Fallback: use overall F1 score distributed evenly across turns (with masking)
        #         for role in agent_roles:
        #             if should_mask:
        #                 reward_tensor_map[f'{role}_turn_level_reward'][i, :] = 0.0
        #             else:
        #                 reward_tensor_map[f'{role}_turn_level_reward'][i, :] = combined_scores[i] / max_num_turns

        reward_tensor_map['evidence'] = evidence
        
        # Aggregate tracking metrics across the batch
        if len(tracking_metrics_list) > 0:
            # Average metrics across all samples in the batch
            avg_memory_size = sum(m.get('memory_size', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_inserts = sum(m.get('memory_insert_count', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_deletes = sum(m.get('memory_delete_count', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_updates = sum(m.get('memory_update_count', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_ops = sum(m.get('memory_operation_count', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_precision = sum(m.get('evidence_precision', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_recall = sum(m.get('evidence_recall', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_rank = sum(m.get('avg_retrieval_rank', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_failure = sum(m.get('memory_failure_rate', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_retrieval_failure = sum(m.get('retrieval_failure_rate', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_total_failure = sum(m.get('total_failure_rate', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_memory_token_count = sum(m.get('memory_token_count', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            avg_compression_ratio = sum(m.get('memory_compression_ratio', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            
            # Add to reward_tensor_map as scalar tensors for logging
            reward_tensor_map['memory_size'] = torch.tensor(avg_memory_size, dtype=torch.float32)
            reward_tensor_map['memory_insert_count'] = torch.tensor(avg_memory_inserts, dtype=torch.float32)
            reward_tensor_map['memory_delete_count'] = torch.tensor(avg_memory_deletes, dtype=torch.float32)
            reward_tensor_map['memory_update_count'] = torch.tensor(avg_memory_updates, dtype=torch.float32)
            reward_tensor_map['memory_ops'] = torch.tensor(avg_memory_ops, dtype=torch.float32)
            reward_tensor_map['evidence_precision'] = torch.tensor(avg_precision, dtype=torch.float32)
            reward_tensor_map['evidence_recall'] = torch.tensor(avg_recall, dtype=torch.float32)
            reward_tensor_map['avg_retrieval_rank'] = torch.tensor(avg_rank, dtype=torch.float32)
            reward_tensor_map['memory_failure_rate'] = torch.tensor(avg_memory_failure, dtype=torch.float32)
            reward_tensor_map['retrieval_failure_rate'] = torch.tensor(avg_retrieval_failure, dtype=torch.float32)
            reward_tensor_map['total_failure_rate'] = torch.tensor(avg_total_failure, dtype=torch.float32)
            reward_tensor_map['memory_token_count'] = torch.tensor(avg_memory_token_count, dtype=torch.float32)
            reward_tensor_map['memory_compression_ratio'] = torch.tensor(avg_compression_ratio, dtype=torch.float32)
        
        # Compute category statistics for all modes (training, validation, test)
        # For training: metrics are reported per-batch (not accumulated)
        # For validation/test: metrics are accumulated across batches
        is_validate = data.meta_info.get('validate', False)
        current_split = data.meta_info.get('split', 'train')
        # print(f"\n[RewardManager] Split={current_split}, validate={is_validate}: Computing per-category statistics...")
        
        # Two-stage aggregation for per-category metrics:
        # Stage 1 (here): Aggregate raw scores from all samples in THIS BATCH
        # Stage 2 (in ray_trainer): Aggregate batch-level stats across MULTIPLE BATCHES
        
        batch_category_stats = {}
        
        # Collect all individual scores for each category across all samples in this batch
        for sample_idx, sample_category_raw in enumerate(category_stats_list):
            # sample_category_raw = {'f1_scores': {cat: [scores]}, 'bleu_scores': {cat: [scores]}}
            for cat in sample_category_raw['f1_scores'].keys():
                if cat not in batch_category_stats:
                    batch_category_stats[cat] = {'f1_scores': [], 'bleu_scores': []}
                # Extend with individual scores from this sample (no averaging yet)
                batch_category_stats[cat]['f1_scores'].extend(sample_category_raw['f1_scores'][cat])
                batch_category_stats[cat]['bleu_scores'].extend(sample_category_raw['bleu_scores'][cat])
        
        # Compute batch-level aggregated statistics (sum + count, not average yet)
        # We compute the average later in the trainer to avoid re-computing sums
        final_category_stats = {}
        for cat, scores_dict in batch_category_stats.items():
            f1_list = scores_dict['f1_scores']
            bleu_list = scores_dict['bleu_scores']
            if len(f1_list) > 0:
                final_category_stats[cat] = {
                    'f1_sum': sum(f1_list),  # Keep as sum, not average
                    'bleu_sum': sum(bleu_list),  # Keep as sum, not average
                    'count': len(f1_list)
                }
                # For logging, compute average
                avg_f1 = final_category_stats[cat]['f1_sum'] / final_category_stats[cat]['count']
                avg_bleu = final_category_stats[cat]['bleu_sum'] / final_category_stats[cat]['count']
                # print(f"[RewardManager] Category {cat}: F1={avg_f1:.4f}, BLEU={avg_bleu:.4f}, count={final_category_stats[cat]['count']}")
        
        # Add per-category metrics to reward_tensor_map as scalar tensors
        if len(final_category_stats) > 0:
            # print(f"[RewardManager] Adding {len(final_category_stats)} category metrics to reward_tensor_map...")
            for cat, stats in final_category_stats.items():
                # Get human-readable category name
                cat_name = CATEGORY_NAMES.get(cat, f'unknown_cat_{cat}')
                # if cat not in CATEGORY_NAMES:
                    # print(f"[RewardManager] Warning: Unknown category ID {cat}, using fallback name")
                
                # Add batch-level sum and count as scalar tensors (not averages)
                # Trainer will compute the global average from these sums
                reward_tensor_map[f'{cat_name}_f1_sum'] = torch.tensor(stats['f1_sum'], dtype=torch.float32)
                reward_tensor_map[f'{cat_name}_bleu_sum'] = torch.tensor(stats['bleu_sum'], dtype=torch.float32)
                reward_tensor_map[f'{cat_name}_count'] = torch.tensor(stats['count'], dtype=torch.float32)
                # For logging
                avg_f1 = stats['f1_sum'] / stats['count']
                avg_bleu = stats['bleu_sum'] / stats['count']
                # print(f"[RewardManager] Added {cat_name}: F1={avg_f1:.4f}, BLEU={avg_bleu:.4f}, count={stats['count']}")
        else:
            # print(f"[RewardManager] No category data found in this batch")
            pass

        # print(f"\n[RewardManager] Turn-level rewards already assigned via dia_id causality (see above). Skipping old reward assignment logic.")
        # Per-role shaping weights:
        # - meta_thinking gets shared + evidence term
        # - reasoning gets shared + (1 - retrieval_failure_rate) term
        fact_evidence_bonus = _float_env("REMA_REWARD_FACT_EVIDENCE_BONUS", 0.1, min_value=0.0)
        retrieval_failure_bonus = _float_env("REMA_REWARD_RETRIEVAL_FAILURE_BONUS", 0.1, min_value=0.0)

        retrieval_failure_rates = [m.get('retrieval_failure_rate', 1.0) for m in tracking_metrics_list]

        # print(f"\n[RewardManager] Processing {len(data)} data items to assign rewards...")
        for i_bsz in range(len(data)):
            data_item = data[i_bsz]  # DataProtoItem

            # Use precomputed combined score (already masked for incomplete trajectories)
            score = combined_scores[i_bsz]
            
            num_turns = data_item.non_tensor_batch['num_turns']
            
            for i_role, role in enumerate(agent_roles):
                
                # Both roles get the same combined score (aligned incentives)
                # Score is already masked in combined_scores computation above
                
                if i_bsz == 0:
                    turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                    # print(f"  - {role}_turn_finished: {turn_finished}")

                # TODO:: Should add my format reward here not this one !

                # if turn_finished == 0 and data_item.meta_info['use_format_reward'] and max_num_turns == 1:
                #     # XXX(ziyu): only add format reward for normally finished 1-turn conversation
                #     last_round_msg = data_item.non_tensor_batch['history'][i_role]
                #     assert last_round_msg['role'] == role, role

                #     format_r = compute_format_r(data_source, role, last_round_msg['content'])
                #     if i_bsz == 0:
                #         print(f"  - Adding format reward for {role}: {format_r}")
                #     score += format_r
                reward_tensor_map[f'{role}_turn_level_reward'][i_bsz, num_turns - 1] = score
                # if i_bsz == 0:
                    # print(f"  - Assigned {role}_turn_level_reward[{i_bsz}, {num_turns - 1}] = {score}")


        # Only cache the best memory during training (test/validation already has only one memory)
        # current_split = data.meta_info['split']
        # if current_split == "train":
        #     # print(f"\n[RewardManager] Training mode: Saving memories (sampling from top {self.top_k_percentage*100:.0f}% by reward)...")
        #     for key, score_memory_list in memory_score_dict.items():
        #         # score_memory_list: list of (score, memory)
        #         if len(score_memory_list) == 0:
        #             # print(f"[RewardManager] No memory found for {key}, skipping save.")
        #             continue
                
        #         # Sort by score in descending order
        #         sorted_list = sorted(score_memory_list, key=lambda x: x[0], reverse=True)
                
        #         # Calculate top k% of memories (at least 1)
        #         num_top_k = max(1, int(len(sorted_list) * self.top_k_percentage))
        #         top_k_candidates = sorted_list[:num_top_k]
                
        #         # Sample one from top k%
        #         selected_score, selected_memory_info = random.choice(top_k_candidates)
                
        #         if selected_memory_info is not None:
        #             # Online learning, save the sampled memory to be used in next batch
        #             memory_manager.cache_snapshot(
        #                 selected_memory_info["memory"], 
        #                 sample_id=selected_memory_info["conv_id"], 
        #                 chunk_id=selected_memory_info["chunk_id"], 
        #                 epoch=selected_memory_info["epoch"], 
        #                 split=selected_memory_info["split"],
        #                 snapshot_suffix=selected_memory_info.get("snapshot_suffix", ""),
        #             )
        #             # print(f"[RewardManager] Saved memory for {key} with score {selected_score:.4f} (sampled from top {num_top_k}/{len(sorted_list)}, best={sorted_list[0][0]:.4f})")
        #         else:
        #             # print(f"[RewardManager] Selected memory is None for {key}, skipping save.")
        #             pass
        # else:
        #     # print(f"\n[RewardManager] Split={current_split}: Skipping memory caching (test/validation are read-only)")
        #     pass

        # Return both reward tensors in a dictionary
        # print(f"\n[RewardManager] Final reward_tensor_map keys: {list(reward_tensor_map.keys())}")
        # for key, tensor in reward_tensor_map.items():
        #     if isinstance(tensor, torch.Tensor):
        #         print(f"[RewardManager] {key} shape: {tensor.shape}, dtype: {tensor.dtype}")
        #         if tensor.numel() > 0:
        #             print(f"[RewardManager] {key} stats - mean: {tensor.mean().item():.4f}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
        #             # Handle 0-dim tensors (scalars) vs 1+dim tensors
        #             if tensor.dim() == 0:
        #                 print(f"[RewardManager] {key} value: {tensor.item()}")
        #             else:
        #                 print(f"[RewardManager] {key} sample values[0]: {tensor[0]}")
        # print("="*80)
        # print("REWARD MANAGER __call__ COMPLETED")
        # print("="*80 + "\n")
        return reward_tensor_map
