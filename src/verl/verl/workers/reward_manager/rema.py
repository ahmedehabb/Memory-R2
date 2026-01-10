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

def compute_score_fn(compute_score, params):
    # data_source, response, ground_truth, extra_info = params
    qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, extra_info = params
    return compute_score(qa_pairs, conv_id, chunk_id, speakers, epoch, split, index, session_time, extra_info)

def locomo_score(qa_pairs: list[dict], conv_id: int, chunk_id: int, speakers: list[str], epoch: int, split: str, index: int, session_time: str, extra_info: dict=None) -> tuple[float, dict]:
    key = f"{conv_id}_chunk{chunk_id}"
    memory = MemoryManager().get_snapshot(sample_id=conv_id, chunk_id=chunk_id, epoch=epoch, split=split, index_in_batch=index)
    
    # Compute score for all QA pairs and return average
    qa_scores = 0.0
    num_questions = len(qa_pairs)

    # Variables to track evidence retrieval
    total_evidences = 0
    total_evidences_retrieved = 0
    
    print(f"[LocomoScore] Processing {num_questions} questions for conv {conv_id}, chunk {chunk_id}")
    
    for qa_idx, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        gold_answer = str(qa_pair['answer']).strip()
        evidence = qa_pair.get('evidence', None)

        # Here we need to ask Question and get answer from response
        # Will use all_dia_ids to check if the retrieved memories are relevant 
        prompt, all_dia_ids = generate_qa_prompt(memory, speaker_1=speakers[0], speaker_2=speakers[1], question=question, session_time=session_time, top_k_per_speaker=20, similarity_threshold=0.1, use_similarity=True)
        response = judge_with_llm(prompt)
        predicted_answer = extract_answer_from_text(response)

        # Use f1 score as the metric
        question_score = compute_f1(predicted_answer, gold_answer)

        if question_score != 1.0:
            print(f"[LocomoScore] Mismatch detected. Gold: {gold_answer}, full response: {response}")
            retrieved_memory = prompt.find("Memories for user")  # Find the start of the memories
            print(f"Retrieved Memory we got:\n{prompt[retrieved_memory:]}\n")

        qa_scores += question_score

        # Now checking that evidence dia_ids are in retrieved dia_ids
        evidences_retrieved = 0
        total_evidences += len(evidence)
        for evidence_dia_id in evidence:
            if evidence_dia_id not in all_dia_ids:
                print(f"[LocomoScore] Warning: Evidence dia_id {evidence_dia_id} not found in retrieved dia_ids {all_dia_ids}")
            else:
                evidences_retrieved += 1
        total_evidences_retrieved += evidences_retrieved
        print(f"[LocomoScore] Retrieved {evidences_retrieved}/{len(evidence)} evidence dia_ids.")


        print(f"[LocomoScore] Q{qa_idx+1}/{num_questions}: {question[:50]}...")
        print(f"[LocomoScore] Gold: {gold_answer}, Predicted: {predicted_answer}, Match: {question_score}")
    
    # Calculate average score
    avg_score = qa_scores / num_questions if num_questions > 0 else 0.0
    print(f"[LocomoScore] Average score: {avg_score:.3f} ({qa_scores}/{num_questions})")

    # Calculate evidence retrieval efficiency (percentage of evidence dia_ids retrieved)
    evidence_retrieval_efficiency = total_evidences_retrieved / total_evidences if total_evidences > 0 else 0.0
    print(f"[LocomoScore] Evidence retrieval efficiency: {evidence_retrieval_efficiency:.3f}")

    # Adjust final score based on evidence retrieval efficiency
    final_score = avg_score + evidence_retrieval_efficiency
    print(f"[LocomoScore] Final adjusted score: {final_score:.3f}")

    memory_info = {
        "key": key,
        "memory": memory,
        "conv_id": conv_id,
        "chunk_id": chunk_id,
        "epoch": epoch,
        "split": split
    }
    # Return both avg_score (QA only) and final_score (QA + evidence reward)
    return avg_score, evidence_retrieval_efficiency, final_score, memory_info

def _rema_math_format_reward_fn(role, response_str):
    if 'boxed' in response_str:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def _rema_laaj_format_reward_fn(role, response_str):
    from verl.utils.reward_score.pairwise_laaj import extract_final_verdict
    ans = extract_final_verdict(response_str)
    if ans is not None:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def compute_format_r(data_source, role, response_str):
    if data_source == "ReMA-math":
        return _rema_math_format_reward_fn(role, response_str)
    elif data_source == 'ReMA-laaj':
        return _rema_laaj_format_reward_fn(role, response_str)
    else:
        raise ValueError(f'Unknown {data_source=} for format reward.')

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
            # data[i].non_tensor_batch['data_source'],
            #  data[i].non_tensor_batch['response'],
            #  data[i].non_tensor_batch['reward_model']['ground_truth'],
             json.loads(data[i].non_tensor_batch['qa_pairs_json']),
             data[i].non_tensor_batch['sample_id'],
             data[i].non_tensor_batch['chunk_id'],
             data[i].non_tensor_batch['speakers'],
             data.meta_info['epoch'],
             data.meta_info['split'],
             data[i].batch['rollout_idx'],  # Use rollout_idx computed AFTER repeating
             data[i].non_tensor_batch['session_time'],
             data[i].non_tensor_batch.get('extra_info', None),
             )
            for i in range(len(data))
        ]
        print(f"[RewardManager] Prepared {len(params)} parameter sets")
        if len(params) > 0:
            print(f"[RewardManager] Sample params[0]:")
            print(f"  - qa_pairs_json: {params[0][0]}")
            print(f"  - extra_info: {params[0][2]}")

        scores = []
        qa_scores = []  # Track QA accuracy separately
        evidence_scores = [] # Track evidence scores separately
        memory_infos = []  # Collect memory info from each result
        print(f"\n[RewardManager] Starting score computation with ProcessPool...")
        with ProcessPool(max_workers=8) as pool:  # Parallel processing with 8 workers
            future = pool.map(partial(compute_score_fn, self.compute_score), params, timeout=300)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                while True:
                    try:
                        result = next(iterator)
                        # New format: (qa_score, evidence format, total_score, memory_info)
                        qa_score, evidence_score, total_score, memory_info = result
                        qa_scores.append(qa_score)
                        evidence_scores.append(evidence_score)
                        scores.append(total_score)
                        memory_infos.append(memory_info)
                    except TimeoutError:
                        print('[RewardManager] Time Out')
                        qa_scores.append(0.0)
                        scores.append(0.0)
                        evidence_scores.append(0.0)
                        memory_infos.append(None)
                    except TimeoutException:
                        print('[RewardManager] Math verify internal timeout')
                        qa_scores.append(0.0)
                        scores.append(0.0)
                        evidence_scores.append(0.0)
                        memory_infos.append(None)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"[RewardManager] Error: {e}")
                        raise e
                    pbar.update(1)
        print(f"[RewardManager] Score computation complete. Got {len(scores)} total scores and {len(qa_scores)} QA scores")
        
        # Build memory_score_dict from collected results
        memory_score_dict = {}
        for i, (score, memory_info) in enumerate(zip(scores, memory_infos)):
            if memory_info is not None and memory_info["memory"] is not None:
                key = memory_info["key"]
                if key not in memory_score_dict:
                    memory_score_dict[key] = []
                memory_score_dict[key].append((score, memory_info))
        print(f"[RewardManager] Built memory_score_dict with {len(memory_score_dict)} unique conversation-chunk pairs")
        
        assert len(scores) == len(data)
        assert len(evidence_scores) == len(data)
        assert len(qa_scores) == len(data)
        accuracy = torch.tensor(qa_scores, dtype=torch.float32) # bsz - QA accuracy only
        print(f"\n[RewardManager] Accuracy tensor (QA only) shape: {accuracy.shape}, dtype: {accuracy.dtype}")
        print(f"[RewardManager] Accuracy stats - mean: {accuracy.mean().item():.4f}, min: {accuracy.min().item():.4f}, max: {accuracy.max().item():.4f}")
        print(f"[RewardManager] Accuracy values: {accuracy[:min(5, len(accuracy))].tolist()}...")
        reward_tensor_map['acc'] = accuracy
        
        print(f"\n[RewardManager] Processing {len(data)} data items to assign rewards...")
        for i_bsz in range(len(data)):
            data_item = data[i_bsz]  # DataProtoItem
            # response_str = data_item.non_tensor_batch['response']
            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # data_source = data_item.non_tensor_batch['data_source']
            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )

            # Instead of using total score, try to separate the scores for each role
            # score = scores[i_bsz]
            qa_score = qa_scores[i_bsz]
            evidence_score = evidence_scores[i_bsz]
            
            num_turns = data_item.non_tensor_batch['num_turns']
            
            for i_role, role in enumerate(agent_roles):
                if i_role == 0:
                    # fact retrieval role gets evidence score
                    score = evidence_score
                elif i_role == 1:
                    # memory manager get qa score 
                    score = qa_score

                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                if i_bsz == 0:
                    print(f"  - {role}_turn_finished: {turn_finished}")
                if data_item.meta_info['mask_unfinished_reward']:
                    # if conversation is not finised normally, i.e. with ['FINISH']
                    #  the reward should be zero.
                    # `turn_finished` is 0 means finished normally.
                    score = score if turn_finished == 0 else 0.0

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

        # Now in the end we save the best memories based on accuracy
        print(f"\n[RewardManager] Saving memories (sampling from top {self.top_k_percentage*100:.0f}% by reward)...")
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

        # Return both reward tensors in a dictionary
        print(f"\n[RewardManager] Final reward_tensor_map keys: {list(reward_tensor_map.keys())}")
        for key, tensor in reward_tensor_map.items():
            if isinstance(tensor, torch.Tensor):
                print(f"[RewardManager] {key} shape: {tensor.shape}, dtype: {tensor.dtype}")
                if tensor.numel() > 0:
                    print(f"[RewardManager] {key} stats - mean: {tensor.mean().item():.4f}, min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
                    print(f"[RewardManager] {key} sample values[0]: {tensor[0]}")
        print("="*80)
        print("REWARD MANAGER __call__ COMPLETED")
        print("="*80 + "\n")
        return reward_tensor_map