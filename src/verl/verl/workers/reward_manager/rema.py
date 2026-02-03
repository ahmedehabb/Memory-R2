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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from typing import Dict

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException
from verl.rema_trainer.memory.utils.parse_response import extract_answer_from_text
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.utils.qa_prompt_generator import generate_qa_prompt
from verl.rema_trainer.memory.judge_llm import judge_with_llm

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
    qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, session_id, session_evidences, extra_info, mem_op_stats = params
    return compute_score(qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, session_id, session_evidences, extra_info, mem_op_stats)

def process_single_qa(qa_pair, memory, speakers, session_time):
    """Process a single QA pair - to be called in parallel"""
    question = qa_pair['question']
    gold_answer = str(qa_pair['answer']).strip()
    evidence = qa_pair.get('evidence', None)
    category = qa_pair.get('category', 0)

    # Generate prompt for memory retrieval (to get dia_ids)
    prompt, speaker_1_dia_ids, speaker_2_dia_ids = generate_qa_prompt(memory, speaker_1=speakers[0], speaker_2=speakers[1], 
                                             question=question, session_time=session_time, 
                                             top_k_per_speaker=20, similarity_threshold=0.0, use_similarity=True)
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

def compute_memory_penalty(mem_op_stats, weights=None):
    """
    Compute a penalty for memory actions during a rollout.

    Args:
        mem_op_stats: dict with keys 'insert_successful', 'delete_successful', 'update_successful'
        weights: dict with weights for each action type, e.g.:
            {"INSERT": 0.3, "UPDATE": 0.1, "DELETE": 0.05}

    Returns:
        penalty: float, higher penalty = worse memory usage
    """
    if weights is None:
        weights = {"INSERT": 0.2, "UPDATE": 0.1, "DELETE": 0.05}

    penalty = (
        mem_op_stats.get('insert_successful', 0) * weights["INSERT"] +
        mem_op_stats.get('update_successful', 0) * weights["UPDATE"] +
        mem_op_stats.get('delete_successful', 0) * weights["DELETE"]
    )
    return penalty

def locomo_score(qa_pairs: list[dict], conv_id: int, chunk_id: int, speakers: list[str], epoch: int, split: str, index: int, session_time: str, session_id: int, session_evidences: list, extra_info: dict=None, mem_op_stats: dict=None) -> tuple[float, dict]:
    key = f"{conv_id}_chunk{chunk_id}_epoch{epoch}"
    memory = MemoryManager().get_snapshot(sample_id=conv_id, chunk_id=chunk_id, epoch=epoch, split=split, index_in_batch=index)
    
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
        'retrieval_failure_rate': 0.0,
    }
    
    # Memory size: number of memory items stored
    if memory is not None and hasattr(memory, 'memories'):
        tracking_metrics['memory_size'] = len(memory.memories)
        print(f"[LocomoScore] Memory size for conv {conv_id}, chunk {chunk_id}: {tracking_metrics['memory_size']} memory items")
    
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
        # print(f"[LocomoScore] Memory operations for conv {conv_id}, chunk {chunk_id}:")
        # print(f"  - Insert: {tracking_metrics['memory_insert_count']}")
        # print(f"  - Delete: {tracking_metrics['memory_delete_count']}")
        # print(f"  - Update: {tracking_metrics['memory_update_count']}")
        # print(f"  - Total operations: {tracking_metrics['memory_operation_count']}")
    
    # Compute score for all QA pairs and return average
    qa_scores = 0.0
    bleu_scores = 0.0
    num_questions = len(qa_pairs)
    
    # For computing retrieval quality metrics
    total_evidence_precision = 0.0
    total_evidence_recall = 0.0
    total_avg_rank = 0.0
    total_retrieval_failures = 0
    num_questions_with_evidence = 0

    # Per-category tracking
    category_f1_scores = {}  # {category: [f1_scores]}
    category_bleu_scores = {}  # {category: [bleu_scores]}

    # Calculate session-level evidence coverage
    # Compare memory's dia_ids_set against session_evidences needed for this session
    if session_evidences and hasattr(memory, 'dia_ids_set'):
        session_evidences_set = set(session_evidences)
        covered_evidences = memory.dia_ids_set.intersection(session_evidences_set)
        evidence_retrieval_coverage = len(covered_evidences) / len(session_evidences_set) if len(session_evidences_set) > 0 else 0.0
        print(f"[LocomoScore] Session {session_id} evidence coverage: {len(covered_evidences)}/{len(session_evidences_set)} ({evidence_retrieval_coverage:.3f})")
        if len(covered_evidences) < len(session_evidences_set):
            missing = session_evidences_set - covered_evidences
            print(f"[LocomoScore] Missing evidence dia_ids: {sorted(list(missing))[:10]}...")  # Show first 10
    else:
        evidence_retrieval_coverage = 0.0
        print(f"[LocomoScore] No session_evidences or dia_ids_set available, evidence coverage = 0.0")
    
    print(f"[LocomoScore] Processing {num_questions} questions for conv {conv_id}, chunk {chunk_id}")
    
    # Handle different question counts
    if num_questions == 0:
        # No questions - empty results will naturally produce zero scores
        results = []
    else:
        # Use parallel processing for any number of questions
        print(f"[LocomoScore] Using parallel processing for {num_questions} questions")
        with ThreadPoolExecutor(max_workers=min(num_questions, 8)) as executor:
            futures = [executor.submit(process_single_qa, qa_pair, memory, speakers, session_time) 
                      for qa_pair in qa_pairs]
            results = [future.result() for future in futures]
    
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

        # Accumulate scores for averaging
        qa_scores += question_score
        bleu_scores += bleu_score
        
        print(f"[LocomoScore] Q{qa_idx+1}/{num_questions} [{CATEGORY_NAMES[category]}]: {result['question']}")
        print(f"[LocomoScore] Gold: {result['gold_answer']}, Predicted: {result['predicted_answer']}, F1: {question_score}, BLEU: {bleu_score}")
        print(f"[LocomoScore] Speaker 1 dia_ids (ranked): {speaker_1_dia_ids}")
        print(f"[LocomoScore] Speaker 2 dia_ids (ranked): {speaker_2_dia_ids}")

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
            
            # Average rank of needed evidence in retrieval results
            ranks = []
            for needed_id in needed_set:
                if needed_id in dia_ids_retrieved_combined:
                    rank = dia_ids_retrieved_combined.index(needed_id) + 1
                    ranks.append(rank)
                else:
                    # Not retrieved - count as retrieval failure
                    total_retrieval_failures += 1
            
            if len(ranks) > 0:
                total_avg_rank += sum(ranks) / len(ranks)
            
            num_questions_with_evidence += 1
        
        if question_score < 0.5:  # Use < instead of != to handle floating point precision
            print(f"[LocomoScore] === Mismatch Analysis (F1={question_score:.3f}) ===")
            
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
            
            print(f"[LocomoScore] Evidence needed: {sorted(list(needed_set))} (total: {len(needed_set)})")
            print(f"[LocomoScore] Speaker 1 retrieved: {speaker_1_dia_ids} (total: {len(speaker_1_dia_ids)})")
            print(f"[LocomoScore] Speaker 2 retrieved: {speaker_2_dia_ids} (total: {len(speaker_2_dia_ids)})")
            print(f"[LocomoScore] Evidence in memory: (total: {len(memory_set)})")
            
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
                print(f"[LocomoScore] Evidence coverage (recall): {coverage:.1%} ({len(correctly_retrieved)}/{len(needed_set)} needed retrieved)")
                print(f"[LocomoScore] Evidence precision: {precision:.1%} ({len(correctly_retrieved)}/{len(retrieved_set)} retrieved were relevant)")
                if len(extra_retrieved) > 0:
                    print(f"[LocomoScore] Extra (irrelevant) retrievals: {len(extra_retrieved)} items not needed")
            
            # Report issues
            if len(not_in_memory_set) > 0:
                print(f"[LocomoScore] MEMORY PROBLEM: {len(not_in_memory_set)}/{len(needed_set)} needed dia_ids not saved in memory")
                print(f"[LocomoScore] Missing from memory: {sorted(list(not_in_memory_set))}")
            
            if len(retrieval_failure_set) > 0:
                print(f"[LocomoScore] RETRIEVAL PROBLEM: {len(retrieval_failure_set)}/{len(in_memory_set)} dia_ids in memory but not retrieved")
                print(f"[LocomoScore] In memory but not retrieved: {sorted(list(retrieval_failure_set))}")
            
            # Show ranking analysis
            if len(needed_set) > 0:
                print(f"[LocomoScore] === Ranking Analysis (F1={question_score:.3f}) ===")
                for needed_id in sorted(list(needed_set)):
                    if needed_id in speaker_1_dia_ids:
                        rank = speaker_1_dia_ids.index(needed_id) + 1
                        print(f"[LocomoScore]   dia_id '{needed_id}' found in Speaker 1 at rank {rank}/{len(speaker_1_dia_ids)}")
                    elif needed_id in speaker_2_dia_ids:
                        rank = speaker_2_dia_ids.index(needed_id) + 1
                        print(f"[LocomoScore]   dia_id '{needed_id}' found in Speaker 2 at rank {rank}/{len(speaker_2_dia_ids)}")
                    else:
                        print(f"[LocomoScore]   dia_id '{needed_id}' NOT RETRIEVED from either speaker")
            
            # Show retrieved memory context
            if 'prompt' in result and result['prompt']:
                retrieved_memory_idx = result['prompt'].find("Memories for user")
                if retrieved_memory_idx != -1:
                    memory_section = result['prompt'][retrieved_memory_idx:]
                    print(f"[LocomoScore] Retrieved memory preview:\n{memory_section}...\n")
    
    # Calculate average scores
    avg_f1_score = qa_scores / num_questions if num_questions > 0 else 0.0
    avg_bleu_score = bleu_scores / num_questions if num_questions > 0 else 0.0
    print(f"[LocomoScore] Average F1 score: {avg_f1_score:.3f} ({qa_scores}/{num_questions})")
    print(f"[LocomoScore] Average BLEU score: {avg_bleu_score:.3f} ({bleu_scores}/{num_questions})")
    print(f"[LocomoScore] Session evidence coverage: {evidence_retrieval_coverage:.3f}")
    
    # Compute average retrieval quality metrics
    if num_questions_with_evidence > 0:
        tracking_metrics['evidence_precision'] = total_evidence_precision / num_questions_with_evidence
        tracking_metrics['evidence_recall'] = total_evidence_recall / num_questions_with_evidence
        tracking_metrics['avg_retrieval_rank'] = total_avg_rank / num_questions_with_evidence
        
        # Retrieval failure rate: % of needed evidence that couldn't be retrieved
        total_needed_evidence = sum(len(qa_pair.get('evidence', [])) for qa_pair in qa_pairs if qa_pair.get('evidence'))
        if total_needed_evidence > 0:
            tracking_metrics['retrieval_failure_rate'] = total_retrieval_failures / total_needed_evidence
        
        print(f"[LocomoScore] Retrieval quality metrics:")
        print(f"  - Evidence precision: {tracking_metrics['evidence_precision']:.3f}")
        print(f"  - Evidence recall: {tracking_metrics['evidence_recall']:.3f}")
        print(f"  - Avg retrieval rank: {tracking_metrics['avg_retrieval_rank']:.1f}")
        print(f"  - Retrieval failure rate: {tracking_metrics['retrieval_failure_rate']:.3f}")

    # Return raw category scores (not averages) for global aggregation
    category_raw_scores = {
        'f1_scores': category_f1_scores,  # {category: [individual_f1_scores]}
        'bleu_scores': category_bleu_scores  # {category: [individual_bleu_scores]}
    }

    memory_info = {
        "key": key,
        "memory": memory,
        "conv_id": conv_id,
        "chunk_id": chunk_id,
        "epoch": epoch,
        "split": split
    }
    
    # Cache is never used, since we always get new stuff due to (generation, retrieval, ... ) randomness
    # Force merge cache before returning (critical for ProcessPool workers)
    # from verl.rema_trainer.memory.judge_llm import merge_to_main_cache
    # merge_to_main_cache()
    
    # Return F1 score, BLEU score, evidence score, category_raw_scores, memory_info, and tracking_metrics
    return avg_f1_score, avg_bleu_score, evidence_retrieval_coverage, category_raw_scores, memory_info, tracking_metrics

class ReMARewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, top_k_percentage=0.3) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score or _default_compute_score
        self.compute_score = locomo_score
        self.top_k_percentage = top_k_percentage  # sample from top k% of memories (e.g., 0.3 = top 30%)

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

    def rank_to_grpo_rewards(self, scores: torch.Tensor):
        """
        Convert rollout scores into GRPO rewards using linear scaling.
        scores: Tensor of shape (N,) for SAME conversation.
        Returns: Tensor of shape (N,) with values in [-1, +1].
        """
        N = scores.numel()
        if N == 1:
            return torch.zeros_like(scores)
        
        # Handle ties: double argsort gives ranks (0 = best, N-1 = worst)
        ranks = torch.argsort(torch.argsort(scores, descending=True)).float()
        
        # Linear scaling: best=+1, worst=-1, linear interpolation in between
        rewards = 1.0 - 2.0 * ranks / (N - 1)
        
        return rewards

    def __call__(self, data: DataProto)-> Dict[str, torch.Tensor]:
        """We will expand this function gradually based on the available datasets"""

        print("\n" + "="*80)
        print("REWARD MANAGER __call__ STARTED")
        print("="*80)
        print(f"[RewardManager] Input data batch size: {len(data)}")
        print(f"[RewardManager] data.batch keys: {list(data.batch.keys())}")
        print(f"[RewardManager] data.non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")
        print(f"[RewardManager] data.meta_info keys: {list(data.meta_info.keys())}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            print("[RewardManager] Found pre-computed rm_scores, returning directly")
            return data.batch['rm_scores']
        
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']
        print(f"[RewardManager] batch_size: {batch_size}, max_num_turns: {max_num_turns}")

        
        agent_roles = data.meta_info['agent_roles']
        print(f"[RewardManager] agent_roles: {agent_roles}")
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) for role in agent_roles
        }
        print(f"[RewardManager] Initialized reward_tensor_map with keys: {list(reward_tensor_map.keys())}")
        for key, tensor in reward_tensor_map.items():
            print(f"[RewardManager] {key} shape: {tensor.shape}")
        
        already_print_data_sources = {}
        memory_manager = MemoryManager()
        
        print(f"\n[RewardManager] Preparing parameters for score computation...")
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
             )
            for i in range(len(data))
        ]

        qa_scores = []  # Track QA accuracy (F1) separately
        bleu_scores = []  # Track BLEU scores separately
        evidence_scores = [] # Track evidence scores separately
        category_stats_list = []  # Track per-category stats
        memory_infos = []  # Collect memory info from each result
        tracking_metrics_list = []  # Track additional metrics (memory size, retrieval quality, etc.)
        print(f"\n[RewardManager] Starting score computation with ProcessPool...")
        with ProcessPool(max_workers=16) as pool:  # Parallel processing with 16 workers
            future = pool.map(partial(compute_score_fn, self.compute_score), params, timeout=7200)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                while True:
                    try:
                        result = next(iterator)
                        # New format: (f1_score, bleu_score, evidence_score, category_stats, memory_info, tracking_metrics)
                        qa_score, bleu_score, evidence_score, category_stats, memory_info, tracking_metrics = result
                        qa_scores.append(qa_score)
                        bleu_scores.append(bleu_score)
                        evidence_scores.append(evidence_score)
                        category_stats_list.append(category_stats)
                        memory_infos.append(memory_info)
                        tracking_metrics_list.append(tracking_metrics)
                    except TimeoutError:
                        print('[RewardManager] Time Out')
                        qa_scores.append(0.0)
                        bleu_scores.append(0.0)
                        evidence_scores.append(0.0)
                        category_stats_list.append({'f1_scores': {}, 'bleu_scores': {}})
                        memory_infos.append(None)
                        tracking_metrics_list.append({})
                    except TimeoutException:
                        print('[RewardManager] Math verify internal timeout')
                        qa_scores.append(0.0)
                        bleu_scores.append(0.0)
                        evidence_scores.append(0.0)
                        category_stats_list.append({'f1_scores': {}, 'bleu_scores': {}})
                        memory_infos.append(None)
                        tracking_metrics_list.append({})
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"[RewardManager] Error: {e}")
                        raise e
                    pbar.update(1)
        print(f"[RewardManager] Score computation complete. Got {len(qa_scores)} F1 scores, {len(bleu_scores)} BLEU scores, and {len(evidence_scores)} evidence scores")
        
        # Compute combined scores once (single place for reward logic)
        # Give 0 score to incomplete trajectories (turn_finished != 1)
        combined_scores = []
        num_incomplete = 0
        for i in range(len(qa_scores)):
            turn_finished = data[i].batch[f'{agent_roles[0]}_turn_finished'].item()
            # Only apply masking if enabled
            if data[i].meta_info['mask_unfinished_reward']:
                if turn_finished == 1:  # Successfully completed
                    score = 1.0 * qa_scores[i] + 0 * evidence_scores[i]
                else:  # Incomplete trajectory - give 0 score
                    score = 0.0
                    num_incomplete += 1
            else:
                # No masking - use score regardless of completion status
                score = 1.0 * qa_scores[i] + 0 * evidence_scores[i]
            combined_scores.append(score)
        print(f"[RewardManager] Combined scores computed: mean={sum(combined_scores)/len(combined_scores):.4f}")
        print(f"[RewardManager] Incomplete trajectories (turn_finished != 1): {num_incomplete}/{len(combined_scores)}")
        
        # Build memory_score_dict from collected results using combined scores
        memory_score_dict = {}
        for i, memory_info in enumerate(memory_infos):
            if memory_info is not None and memory_info["memory"] is not None:
                key = memory_info["key"]
                if key not in memory_score_dict:
                    memory_score_dict[key] = []
                memory_score_dict[key].append((combined_scores[i], memory_info))
        print(f"[RewardManager] Built memory_score_dict with {len(memory_score_dict)} unique conversation-chunk pairs")
        
        assert len(evidence_scores) == len(data)
        assert len(qa_scores) == len(data)
        assert len(bleu_scores) == len(data)
        assert len(category_stats_list) == len(data)
        accuracy = torch.tensor(qa_scores, dtype=torch.float32) # bsz - F1 accuracy
        bleu = torch.tensor(bleu_scores, dtype=torch.float32) # bsz - BLEU scores
        evidence = torch.tensor(evidence_scores, dtype=torch.float32) # bsz - evidence coverage scores
        reward_tensor_map['acc'] = accuracy
        reward_tensor_map['bleu'] = bleu
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
            avg_failure_rate = sum(m.get('retrieval_failure_rate', 0) for m in tracking_metrics_list) / len(tracking_metrics_list)
            
            # Add to reward_tensor_map as scalar tensors for logging
            reward_tensor_map['memory_size'] = torch.tensor(avg_memory_size, dtype=torch.float32)
            reward_tensor_map['memory_insert_count'] = torch.tensor(avg_memory_inserts, dtype=torch.float32)
            reward_tensor_map['memory_delete_count'] = torch.tensor(avg_memory_deletes, dtype=torch.float32)
            reward_tensor_map['memory_update_count'] = torch.tensor(avg_memory_updates, dtype=torch.float32)
            reward_tensor_map['memory_ops'] = torch.tensor(avg_memory_ops, dtype=torch.float32)
            reward_tensor_map['evidence_precision'] = torch.tensor(avg_precision, dtype=torch.float32)
            reward_tensor_map['evidence_recall'] = torch.tensor(avg_recall, dtype=torch.float32)
            reward_tensor_map['avg_retrieval_rank'] = torch.tensor(avg_rank, dtype=torch.float32)
            reward_tensor_map['retrieval_failure_rate'] = torch.tensor(avg_failure_rate, dtype=torch.float32)

        
        # Compute category statistics for all modes (training, validation, test)
        # For training: metrics are reported per-batch (not accumulated)
        # For validation/test: metrics are accumulated across batches
        is_validate = data.meta_info.get('validate', False)
        current_split = data.meta_info.get('split', 'train')
        print(f"\n[RewardManager] Split={current_split}, validate={is_validate}: Computing per-category statistics...")
        
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
                print(f"[RewardManager] Category {cat}: F1={avg_f1:.4f}, BLEU={avg_bleu:.4f}, count={final_category_stats[cat]['count']}")
        
        # Add per-category metrics to reward_tensor_map as scalar tensors
        if len(final_category_stats) > 0:
            print(f"[RewardManager] Adding {len(final_category_stats)} category metrics to reward_tensor_map...")
            for cat, stats in final_category_stats.items():
                # Get human-readable category name
                cat_name = CATEGORY_NAMES.get(cat, f'unknown_cat_{cat}')
                if cat not in CATEGORY_NAMES:
                    print(f"[RewardManager] Warning: Unknown category ID {cat}, using fallback name")
                
                # Add batch-level sum and count as scalar tensors (not averages)
                # Trainer will compute the global average from these sums
                reward_tensor_map[f'{cat_name}_f1_sum'] = torch.tensor(stats['f1_sum'], dtype=torch.float32)
                reward_tensor_map[f'{cat_name}_bleu_sum'] = torch.tensor(stats['bleu_sum'], dtype=torch.float32)
                reward_tensor_map[f'{cat_name}_count'] = torch.tensor(stats['count'], dtype=torch.float32)
                # For logging
                avg_f1 = stats['f1_sum'] / stats['count']
                avg_bleu = stats['bleu_sum'] / stats['count']
                print(f"[RewardManager] Added {cat_name}: F1={avg_f1:.4f}, BLEU={avg_bleu:.4f}, count={stats['count']}")
        else:
            print(f"[RewardManager] No category data found in this batch")

        print(f"\n[RewardManager] Processing {len(data)} data items to assign rewards...")
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
                    print(f"  - {role}_turn_finished: {turn_finished}")

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
                if i_bsz == 0:
                    print(f"  - Assigned {role}_turn_level_reward[{i_bsz}, {num_turns - 1}] = {score}")

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     prompt_str = data_item.non_tensor_batch['question']
            #     padded_history = data_item.non_tensor_batch['history']
            #     history = padded_history[:num_turns * 2]
            #     already_print_data_sources[data_source] += 1
            #     print("[question]", prompt_str)
            #     print("[ground_truth]", ground_truth)
            #     print("[answer]", response_str)
            #     print("[score]", score)
            #     print("[history]", history)

        # Only cache the best memory during training (test/validation already has only one memory)
        current_split = data.meta_info['split']
        if current_split == "train":
            print(f"\n[RewardManager] Training mode: Saving memories (sampling from top {self.top_k_percentage*100:.0f}% by reward)...")
            for key, score_memory_list in memory_score_dict.items():
                # score_memory_list: list of (score, memory)
                if len(score_memory_list) == 0:
                    print(f"[RewardManager] No memory found for {key}, skipping save.")
                    continue
                
                # Sort by score in descending order
                sorted_list = sorted(score_memory_list, key=lambda x: x[0], reverse=True)
                
                # Calculate top k% of memories (at least 1)
                num_top_k = max(1, int(len(sorted_list) * self.top_k_percentage))
                top_k_candidates = sorted_list[:num_top_k]
                
                # Sample one from top k%
                selected_score, selected_memory_info = random.choice(top_k_candidates)
                
                if selected_memory_info is not None:
                    # Online learning, save the sampled memory to be used in next batch
                    memory_manager.cache_snapshot(
                        selected_memory_info["memory"], 
                        sample_id=selected_memory_info["conv_id"], 
                        chunk_id=selected_memory_info["chunk_id"], 
                        epoch=selected_memory_info["epoch"], 
                        split=selected_memory_info["split"]
                    )
                    print(f"[RewardManager] Saved memory for {key} with score {selected_score:.4f} (sampled from top {num_top_k}/{len(sorted_list)}, best={sorted_list[0][0]:.4f})")
                else:
                    print(f"[RewardManager] Selected memory is None for {key}, skipping save.")
        else:
            print(f"\n[RewardManager] Split={current_split}: Skipping memory caching (test/validation are read-only)")

        # Return both reward tensors in a dictionary
        print(f"\n[RewardManager] Final reward_tensor_map keys: {list(reward_tensor_map.keys())}")
        for key, tensor in reward_tensor_map.items():
            if isinstance(tensor, torch.Tensor):
                print(f"[RewardManager] {key} shape: {tensor.shape}, dtype: {tensor.dtype}")
                if tensor.numel() > 0:
                    print(f"[RewardManager] {key} stats - mean: {tensor.mean().item():.4f}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
                    # Handle 0-dim tensors (scalars) vs 1+dim tensors
                    if tensor.dim() == 0:
                        print(f"[RewardManager] {key} value: {tensor.item()}")
                    else:
                        print(f"[RewardManager] {key} sample values[0]: {tensor[0]}")
        print("="*80)
        print("REWARD MANAGER __call__ COMPLETED")
        print("="*80 + "\n")
        return reward_tensor_map