import os
import yaml
import time
import torch
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from verl import DataProto
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.memory_core.prompt_generator import generate_memory_prompt, generate_judge_prompt, generate_memory_judge_prompt
from verl.rema_trainer.memory.utils.parse_response import extract_llm_json_from_response, extract_answer_from_text
from verl.rema_trainer.memory.utils.qa_prompt_generator import generate_qa_prompt
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.rema_trainer.memory.memory_core.utils import count_tokens
from verl.rema_trainer.memory.judge_llm import judge_with_llm, judge_with_llm_batch
from verl.rema_trainer.memory.teacher_model import TeacherModel
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MemoryGenerationConfig:
    max_prompt_length: int
    max_response_length: int
    truncation: str = 'error'  # 'error', 'left', 'right', 'middle'
    num_gpus: int = 1
    num_rollouts: int = 1
    # Separate top_k for different purposes:
    top_k_memories_for_operations: int = 20  # Memories to show when deciding what to update/delete (total across all turns)
    top_k_memories_for_qa: int = 40  # Memories for QA retrieval (must be divisible by 2 for two speakers)
    similarity_threshold: float = 0.1  # Similarity threshold for memory search (was 0.3)
    use_llm_judge: bool = True  # Whether to use LLM-as-a-judge for answer evaluation (vs simple string matching)
    format_reward_weight: float = 0.2  # Weight for format reward
    answer_reward_weight: float = 1.0  # Weight for answer reward
    memory_reward_weight: float = 0.5  # Weight for memory operation reward
    max_qas_per_chunk: int = 3  # Maximum number of QAs to sample per chunk (for variable-length training)
    
    # Max response length per generation call = max total response length // (max_qas_per_chunk + 1(memory))
    # Same for prompt length
    max_response_length_per_turn: int = field(init=False)
    max_prompt_length_per_turn: int = field(init=False)
    def __post_init__(self):
        self.max_response_length_per_turn = self.max_response_length // (self.max_qas_per_chunk + 1)
        self.max_prompt_length_per_turn = self.max_prompt_length // (self.max_qas_per_chunk + 1)

class MemoryGenerationManager:
    """Generation manager for memory agent that processes chunks and performs memory operations."""
    def __init__(self, tokenizer, actor_rollout_wg, split: str = 'train', config: MemoryGenerationConfig = None):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.split = split
        self.config = config
        assert self.config.top_k_memories_for_qa % 2 == 0, "top_k_memories_for_qa must be divisible by 2 for two speakers"

    def _generate_with_gpu_padding(self, active_batch: DataProto, actor) -> DataProto:
        """
        Generate sequences with automatic GPU padding to ensure batch is divisible by num_gpus.
        
        Args:
            active_batch: DataProto with input_ids, attention_mask, position_ids
            
        Returns:
            DataProto with generated sequences (padding removed)
        """
        if actor is None:
            raise ValueError("Actor must be provided for generation.")
        num_gpus = self.config.num_gpus
        batch_size = active_batch.batch['input_ids'].shape[0]
        # print(f"\n[DEBUG _generate_with_gpu_padding] Batch size: {batch_size}, Num GPUs: {num_gpus}")
        # print(f"[DEBUG _generate_with_gpu_padding] Input tensor dtypes: {[(k, v.dtype) for k, v in active_batch.batch.items()]}")
        if num_gpus <= 1:
            # print("[DEBUG _generate_with_gpu_padding] Single GPU mode - no padding needed")
            return actor.generate_sequences(active_batch)

        remainder = batch_size % num_gpus
        
        if remainder == 0:
            # Already divisible, no padding needed - use batch as-is
            # print(f"[DEBUG _generate_with_gpu_padding] Batch size {batch_size} is divisible by {num_gpus} - no padding")
            return actor.generate_sequences(active_batch)
        
        # Pad by duplicating first sequences
        padding_size = num_gpus - remainder
        # print(f"[DEBUG _generate_with_gpu_padding] Padding batch from {batch_size} to {batch_size + padding_size}")

        # Duplicate first samples to pad
        padded_batch = {}
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
        padded_active_batch = DataProto.from_dict(padded_batch)
        # Copy meta_info from original batch to ensure eos_token_id, pad_token_id etc. are available
        padded_active_batch.meta_info = active_batch.meta_info.copy() if hasattr(active_batch, 'meta_info') else {}

        # print(f"[DEBUG _generate_with_gpu_padding] Padded tensor dtypes: {[(k, v.dtype) for k, v in padded_active_batch.batch.items()]}")
        padded_result = actor.generate_sequences(padded_active_batch)

        # Remove padding from result
        result_dict = {}
        for k, v in padded_result.batch.items():
            result_dict[k] = v[:batch_size]

        result = DataProto.from_dict(result_dict)
        result.meta_info = padded_result.meta_info
        # print(f"[DEBUG _generate_with_gpu_padding] Removed padding, final batch size: {batch_size}")
        return result

    def _evaluate_memory_operations(self, formatted_turns, formatted_memory, memory_operations) -> list[float]:
        """Evaluate memory operations and compute rewards based on their success.
        
        Args:
            formatted_turns: List of dicts representing the formatted conversation turns.
            formatted_memory: List of dicts representing the current memory entries.
            memory_operations: List of dicts representing the operations to evaluate.

        Returns:
            list[float]: List of rewards for each operation.
        """

        # Generate judge prompts for all QA pairs
        judge_prompts = []
        for turns, memory, ops in zip(formatted_turns, formatted_memory, memory_operations):
            judge_prompt = generate_memory_judge_prompt(turns, memory, ops)
            judge_prompts.append(judge_prompt)

        scores = []
        analysis = []

        # --- Batched Gemini judge evaluation ---
        judge_responses = judge_with_llm_batch(judge_prompts)
        
        for i, judge_response_text in enumerate(judge_responses):
            # Parse JSON response
            try:
                judge_json = extract_llm_json_from_response(judge_response_text)
                if not judge_json.get("_parse_success", False):
                    scores.append(0.0)
                    analysis.append("")
                    continue
            except Exception:
                scores.append(0.0)
                analysis.append("")
                continue

            score = float(judge_json.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # clamp
            scores.append(score)
            analysis.append(judge_json.get("analysis", ""))

            print(f"[DEBUG _evaluate_memory_operations] Operations:\n{memory_operations[i]}\n Judge Response:\n{judge_response_text}")
        return scores

    def _evaluate_answers_with_judge_batched(self, all_qa_data: list[dict], use_llm_judge: bool = False) -> list[float]:
        """Use LLM as a judge to evaluate answer quality for multiple QA pairs at once.
        
        This is isolated evaluation - does NOT affect training sequences or input_ids.
        The judge provides continuous scores between 0.0 and 1.0 based on correctness and completeness.
        
        Args:
            all_qa_data: List of dicts, each containing:
                - 'question': The question that was asked
                - 'gold_answer': The reference/gold answer
                - 'predicted_answer': The model's predicted answer
                - Additional metadata (batch_idx, question_idx, conv_id, chunk_id) for logging
            use_llm_judge: If True, use LLM-as-a-judge for evaluation. If False, use simple string matching.
            
        Returns:
            list[float]: Scores between 0.0 and 1.0 for each QA pair (same order as input)
        """
        if not all_qa_data:
            return []
        
        scores = []
        
        # Simple string matching evaluation (no LLM judge)
        if not use_llm_judge:
            for qa_item in all_qa_data:
                gold_answer = qa_item['gold_answer']
                predicted_answer = qa_item['predicted_answer']
                
                # remove . at the end of answers for comparison (do this BEFORE unknown check)
                gold_answer = gold_answer.rstrip('.')
                predicted_answer = predicted_answer.rstrip('.')
                
                # Handle "unknown" cases (check after stripping)
                gold_is_unknown = gold_answer.lower() == "unknown"
                pred_is_unknown = predicted_answer.lower() == "unknown"

                if gold_is_unknown or pred_is_unknown:
                    # Only score 1.0 if both are unknown, otherwise 0.0
                    if gold_is_unknown and pred_is_unknown:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                    continue
                
                # Simple string matching: exact match
                if predicted_answer.lower() == gold_answer.lower():
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
            return scores
        
        # LLM-as-a-judge evaluation
        # Generate judge prompts for all QA pairs
        judge_prompts = []
        for qa_item in all_qa_data:
            judge_prompt = generate_judge_prompt(
                question=qa_item['question'],
                gold_answer=qa_item['gold_answer'],
                predicted_answer=qa_item['predicted_answer']
            )
            judge_prompts.append(judge_prompt)

        # Batched judge evaluation
        judge_responses = judge_with_llm_batch(judge_prompts)
        
        scores = []
        for idx, qa_item in enumerate(all_qa_data):
            judge_response_text = judge_responses[idx].strip()

            # Unknown handling
            gold_unknown = qa_item["gold_answer"].lower() == "unknown"
            pred_unknown = qa_item["predicted_answer"].lower() == "unknown"

            if gold_unknown or pred_unknown:
                scores.append(1.0 if (gold_unknown and pred_unknown) else 0.0)
                continue

            # Parse score
            try:
                score = float(judge_response_text)
                score = max(0.0, min(1.0, score))
            except:
                import re
                nums = re.findall(r"[0-9]*\.?[0-9]+", judge_response_text)
                if nums:
                    score = float(nums[0])
                    score = max(0.0, min(1.0, score))
                else:
                    # fallback binary to exact match
                    pa = qa_item['predicted_answer'].lower()
                    ga = qa_item['gold_answer'].lower()
                    score = 1.0 if (pa == ga) else 0.0

            scores.append(score)
        return scores

    def _get_cache_dir_for_split(self, split: str) -> str:
        """Get cache directory for a given split.
        
        Args:
            split: Dataset split ('train' or 'validation')
            
        Returns:
            str: Cache directory path
        """
        if split == "train":
            return os.getenv("MEMORY_CACHE_DIR", "./memory_cache")
        elif split == "validation":
            return os.getenv("MEMORY_CACHE_DIR_VAL", "./memory_cache_val")
        else:
            raise ValueError(f"Unknown split: {split}")

    def _group_dataset_by_conversation(self, dataset):
        """Group dataset items by conversation ID and sort by chunk ID.
        
        Args:
            dataset: Dataset containing conversation chunks.
            
        Returns:
            dict: {conv_id: [sorted_items]}
        """
        conv_dict = {}
        for item in dataset:
            conv_id = item.get("sample_id", None)
            if conv_id is None:
                raise ValueError("Each dataset item must have a 'sample_id' as conversation id.")
            
            if conv_id not in conv_dict:
                conv_dict[conv_id] = []
            conv_dict[conv_id].append(item)
        
        # Sort items by chunk_id for each conversation
        for conv_id in conv_dict:
            conv_dict[conv_id] = sorted(conv_dict[conv_id], key=lambda x: x.get("chunk_id", -1))
        
        return conv_dict

    def _check_all_snapshots_cached(self, conv_dict, epoch: int):
        """Check if all memory snapshots are already cached for this epoch.
        
        Returns per-split status to enable split-aware caching.
        
        Args:
            conv_dict: Dictionary of conversations {conv_id: [items]}
            epoch: Current epoch number.
            
        Returns:
            dict: {split: bool} - True if all snapshots for that split are cached
        """
        split_status = {}
        
        for conv_id, items in conv_dict.items():
            for item in items:
                chunk_id = item.get("chunk_id", None)
                # Get split from item if available, otherwise use self.split
                split = item.get('_split', self.split)
                cache_dir = self._get_cache_dir_for_split(split)
                
                # Initialize split status if not seen before
                if split not in split_status:
                    split_status[split] = True
                
                # Path format: {MEMORY_CACHE_DIR}/epoch_{epoch}/{sample_id}/chunk_{chunk_id}
                # Check for both pickle and json formats
                cache_file_pkl = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.pkl")
                cache_file_json = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.json")
                if not (os.path.exists(cache_file_pkl) or os.path.exists(cache_file_json)):
                    split_status[split] = False
        
        return split_status

    def _find_resume_chunk_index(self, conv_dict, epoch: int):
        """Find the chunk index to resume from by checking which snapshots exist.
        
        Returns per-split resume indices to enable split-aware resumption.
        
        Args:
            conv_dict: Dictionary of conversations {conv_id: [items]}
            epoch: Current epoch number.
            
        Returns:
            dict: {split: int} - Chunk index to resume from for each split (0 if starting fresh)
        """
        max_chunks = max(len(items) for items in conv_dict.values())
        split_resume_indices = {}
        
        # Check each chunk index to find where we left off per split
        for chunk_idx in range(max_chunks):
            # Check if ALL conversations that have this chunk are cached, per split
            split_cached_status = {}
            
            for conv_id, items in conv_dict.items():
                if chunk_idx < len(items):  # This conversation has this chunk
                    chunk_id = items[chunk_idx].get("chunk_id", None)
                    # Get split from item if available, otherwise use self.split
                    split = items[chunk_idx].get('_split', self.split)
                    cache_dir = self._get_cache_dir_for_split(split)
                    
                    # Initialize split status if not seen before
                    if split not in split_cached_status:
                        split_cached_status[split] = True
                    
                    # Path format: {MEMORY_CACHE_DIR}/epoch_{epoch}/{sample_id}/chunk_{chunk_id}
                    # Check for both pickle and json formats
                    cache_file_pkl = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.pkl")
                    cache_file_json = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.json")
                    if not (os.path.exists(cache_file_pkl) or os.path.exists(cache_file_json)):
                        split_cached_status[split] = False
            
            # Record resume index for each split that's not fully cached at this chunk
            for split, is_cached in split_cached_status.items():
                if not is_cached and split not in split_resume_indices:
                    split_resume_indices[split] = chunk_idx
        
        # For splits that are fully cached, set resume index to max_chunks
        for conv_id, items in conv_dict.items():
            for item in items:
                split = item.get('_split', self.split)
                if split not in split_resume_indices:
                    split_resume_indices[split] = max_chunks
        
        return split_resume_indices

    def _load_cached_memory_states(
        self, 
        conv_dict, 
        conv_managers: Dict[str, MemoryManager], 
        split_resume_indices: Dict[str, int], 
        epoch: int
    ) -> Dict[str, Memory]:
        """Load cached memory states for all conversations up to resume point per split.
        
        Args:
            conv_dict: Dictionary of conversations {conv_id: [items]}
            conv_managers: Dictionary of memory managers {conv_id: MemoryManager}
            split_resume_indices: Dict mapping split to chunk index we're resuming from
            epoch: Current epoch number.
            
        Returns:
            dict: {conv_id: Memory} - Loaded memory states.
        """
        conv_memories: Dict[str, Memory] = {}
        
        # Load the memory state from the previous chunk per split
        for conv_id, items in conv_dict.items():
            # Get the split from the first item of this conversation
            first_item_split = items[0].get('_split', self.split)
            resume_chunk_idx = split_resume_indices.get(first_item_split, 0)
            
            if resume_chunk_idx == 0:
                # Starting fresh for this conversation
                conv_memories[conv_id] = Memory()
                continue
            
            # Find the last cached chunk for this conversation
            last_chunk_idx = min(resume_chunk_idx - 1, len(items) - 1)
            
            if last_chunk_idx >= 0:
                chunk_id = items[last_chunk_idx].get("chunk_id", None)
                # Get split from item if available, otherwise use self.split
                split = items[last_chunk_idx].get('_split', self.split)
                loaded_memory = conv_managers[conv_id].get_snapshot(conv_id, chunk_id, epoch, split)
                if loaded_memory is not None:
                    conv_memories[conv_id] = loaded_memory
                    print(f"Loaded cached memory for conv {conv_id} from chunk {chunk_id} ({split})")
                else:
                    # Cache might be missing, start fresh
                    conv_memories[conv_id] = Memory()
                    print(f"Warning: No cached memory found for conv {conv_id}, chunk {chunk_id}. Starting fresh.")
            else:
                conv_memories[conv_id] = Memory()
        
        return conv_memories

    def _collect_batch_data_for_chunk(
        self, 
        conv_dict, 
        conv_memories: Dict[str, Memory], 
        conv_managers: Dict[str, MemoryManager], 
        chunk_idx: int
    ):
        """Collect batch data for all conversations at a given chunk index.
        
        Args:
            conv_dict: Dictionary of conversations {conv_id: [items]}
            conv_memories: Dictionary of memory states {conv_id: Memory}
            conv_managers: Dictionary of memory managers {conv_id: MemoryManager}
            chunk_idx: Current chunk index to process.
            
        Returns:
            list: Batch data with conv_id, chunk_id, turns, memory, manager, split for each conversation.
        """
        batch_data = []
        
        for conv_id, items in conv_dict.items():
            if chunk_idx < len(items):  # This conversation has this chunk
                item = items[chunk_idx]
                chunk_id = item.get("chunk_id", None)
                turns = item.get("turns_json", None)
                
                if chunk_id is None or turns is None:
                    raise ValueError(f"Item for conv_id {conv_id} must have 'chunk_id' and 'turns_json' fields.")
                
                # Parse turns json if stored as string
                if isinstance(turns, str):
                    turns = json.loads(turns)
                
                # Get split from item if available, otherwise use self.split
                split = item.get('_split', self.split)
                
                batch_data.append({
                    'conv_id': conv_id,
                    'chunk_id': chunk_id,
                    'turns': turns,
                    'memory': conv_memories[conv_id],
                    'manager': conv_managers[conv_id],
                    'split': split,
                })
        
        return batch_data

    def _generate_and_tokenize_memory_prompts(self, batch_data):
        """Generate and tokenize prompts for all conversations in batch.
        
        Args:
            batch_data: List of dicts with conv_id, chunk_id, turns, memory, manager.
            
        Returns:
            tuple: (batch_input_ids, batch_attention_mask, batch_position_ids)
        """
        prompts = []
        formatted_turns_list = []
        formatted_memory_list = []
        
        for data in batch_data:
            prompt, formatted_turns, formatted_memory = generate_memory_prompt(
                data['memory'], 
                data['turns'], 
                top_k_memories=self.config.top_k_memories_for_operations, 
                similarity_threshold=self.config.similarity_threshold, 
                use_similarity=True
            )
            prompts.append(prompt)
            formatted_turns_list.append(formatted_turns)
            formatted_memory_list.append(formatted_memory)
        batch_input_ids, batch_attention_mask, batch_position_ids = self._generate_and_tokenize_prompts(prompts)
        return batch_input_ids, batch_attention_mask, batch_position_ids, formatted_turns_list, formatted_memory_list

    def _generate_and_tokenize_prompts(self, prompts):
        """Generate and tokenize prompts for all conversations in batch.
        
        Args:
            prompts: List of prompt strings.
            
        Returns:
            tuple: (batch_input_ids, batch_attention_mask, batch_position_ids)
        """
        
        # print first 5 prompts for debugging
        # for i, p in enumerate(prompts[:5]):
        #     print(f"Prompt {i}: {p}")

        # Tokenize all prompts
        all_input_ids = []
        all_attention_masks = []
        all_position_ids = []
        
        for prompt in prompts:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_length=self.config.max_prompt_length_per_turn,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.truncation if hasattr(self.config, 'truncation') else 'error',
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            
            all_input_ids.append(input_ids[0])
            all_attention_masks.append(attention_mask[0])
            all_position_ids.append(position_ids[0])
        
        # Stack into batch tensors
        batch_input_ids = torch.stack(all_input_ids, dim=0)
        batch_attention_mask = torch.stack(all_attention_masks, dim=0)
        batch_position_ids = torch.stack(all_position_ids, dim=0)

        # print("generate and tokenize prompts function:")
        # print("batch_input_ids", batch_input_ids, batch_input_ids.shape)
        # print("batch_attention_mask", batch_attention_mask, batch_attention_mask.shape)
        # print("batch_position_ids", batch_position_ids, batch_position_ids.shape)
        # print("each input size", batch_input_ids.shape[1])
        
        return batch_input_ids, batch_attention_mask, batch_position_ids

    def _execute_memory_operations(
        self,
        batch_data,
        response_batch: DataProto,
        total_prompt_length: int
    ) -> tuple[list[float], list[list[dict]]]:
        """Execute memory operations from batch responses without caching.
        
        Args:
            batch_data: List of dicts with conv_id, chunk_id, turns, memory, manager.
            response_batch: DataProto containing generated responses.
            total_prompt_length: Length of prompts (for slicing responses).

        Returns:
            list: Rewards for each conversation based on operation execution success.
                  Reward = json_correctness * operation_success_rate
                  - If JSON invalid: reward = 0.0
                  - If JSON valid but 0 ops: reward = 1.0 * 1.0 = 1.0 (intentional no-ops)
                  - If JSON valid with N ops, M successful: reward = 1.0 * (M/N)
            memory_operations_per_sample: List of lists of dicts - recorded operations per sample.
        """
        rewards = []
        # Per-sample recorded operations (parsed from LLM)
        memory_operations_per_sample: list[list[dict]] = [[] for _ in range(len(batch_data))]
        
        for idx, data in enumerate(batch_data):
            # Extract response for this conversation
            response_ids = response_batch.batch['input_ids'][idx, total_prompt_length:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            # print("response text:", response_text)

            # Parse operations from response
            response_json = extract_llm_json_from_response(response_text)
            json_parse_success = response_json.get("_parse_success", False)
            operations = response_json.get("operations", [])
            
            # If JSON parsing failed, reward is 0
            if not json_parse_success:
                rewards.append(0.0)
                print(f"Conv {data['conv_id']}, chunk {data['chunk_id']}: "
                      f"JSON parsing FAILED - Reward=0.0")
                continue
            
            # Attach turn metadata
            operations = data['manager'].attach_turn_metadata_to_operations(
                operations, data['turns'], data['conv_id']
            )

            # Record the parsed operations for downstream evaluation / judging
            memory_operations_per_sample[idx] = operations # list of dicts

            # Execute operations (even if empty list)
            result = data['manager'].execute_batch(data['memory'], operations)
            
            total_ops = result.get("total_commands", 0)
            successful_ops = result.get("successful", 0)
            
            # Calculate operation success rate
            if total_ops == 0:
                # No operations - intentional, so 100% success
                ops_reward = 1.0
            else:
                ops_reward = successful_ops / total_ops
            
            # Final reward: JSON correct (1.0) * operation success rate
            final_reward = 1.0 * ops_reward
            rewards.append(final_reward)
            
            print(f"Conv {data['conv_id']}, chunk {data['chunk_id']}: "
                  f"JSON=OK, Ops={successful_ops}/{total_ops}, Format Reward={final_reward:.3f}")
            
            if result["status"] not in ["success", "partial"]:
                print(f"Warning: Memory operations had issues: {result}")
        
        return rewards, memory_operations_per_sample

    def generate_memory_snapshots(self, dataset, epoch: int) -> None:
        """Generate memory snapshots for each conversation in the dataset if not cached.
        
        DEPRECATION WARNING: This function should ONLY be used for:
        1. Initial supervised pre-training of memory operations
        2. Generating validation snapshots (not used for training)
        3. Warm-start initialization before RL training begins
        
        ❌ DO NOT USE for RL training loops - this creates off-policy leakage!
        
        For RL training, memory states MUST be generated on-the-fly during training with the current
        policy using run_chunks(). Pre-generating snapshots at epoch start creates a temporal mismatch
        where the agent trains on memory states from an older policy version.
        
        This function processes conversations in parallel by batching chunks at the same position across
        different conversations (all chunk_0s together, then all chunk_1s together, etc.).
        
        Supports resuming from interruptions - if generation stopped at chunk i, it will resume from there.
        Supports combined datasets with split information: if items have '_split' field, it will be used
        to determine which cache directory to use for each item.

        Args:
            dataset: The dataset containing conversations. Items may optionally have '_split' field.
            epoch: The current epoch number.
        """
        # Create cache directories for both splits in case we have a combined dataset
        train_cache_dir = self._get_cache_dir_for_split("train")
        val_cache_dir = self._get_cache_dir_for_split("validation")
        os.makedirs(train_cache_dir, exist_ok=True)
        os.makedirs(val_cache_dir, exist_ok=True)

        # Group dataset by conversation and check cache
        conv_dict = self._group_dataset_by_conversation(dataset)
        
        # Check if this is a combined dataset by looking for _split field
        has_split_info = any('_split' in item for items in conv_dict.values() for item in items)
        split_info = " (combined)" if has_split_info else f" ({self.split})"
        
        # Check cache status per split
        split_cached_status = self._check_all_snapshots_cached(conv_dict, epoch)
        
        # If all splits are fully cached, skip generation entirely
        if all(split_cached_status.values()):
            cached_splits = ', '.join(split_cached_status.keys())
            print(f"✓ All memory snapshots already cached for epoch {epoch} ({cached_splits}). Skipping generation.")
            return
        
        # Find where to resume from per split
        split_resume_indices = self._find_resume_chunk_index(conv_dict, epoch)
        
        # Print status for each split
        for split, is_cached in split_cached_status.items():
            resume_idx = split_resume_indices.get(split, 0)
            if is_cached:
                print(f"✓ Split '{split}': All snapshots cached for epoch {epoch}")
            elif resume_idx > 0:
                print(f"Split '{split}': Resuming from chunk index {resume_idx} for epoch {epoch}")
            else:
                print(f"Split '{split}': Generating snapshots from scratch for epoch {epoch}")
        
        # Initialize memory managers
        conv_managers: Dict[str, MemoryManager] = {conv_id: MemoryManager() for conv_id in conv_dict}
        
        # Load cached memory states up to resume point per split
        conv_memories: Dict[str, Memory] = self._load_cached_memory_states(conv_dict, conv_managers, split_resume_indices, epoch)
        
        # Find maximum number of chunks across all conversations
        max_chunks = max(len(items) for items in conv_dict.values())
        
        # Process chunks starting from resume point, filtering by split
        for chunk_idx in range(max_chunks):
            # Collect all conversations that have this chunk AND need processing (split not fully cached)
            batch_data = []
            
            for conv_id, items in conv_dict.items():
                if chunk_idx < len(items):  # This conversation has this chunk
                    item = items[chunk_idx]
                    chunk_id = item.get("chunk_id", None)
                    turns = item.get("turns_json", None)
                    split = item.get('_split', self.split)
                    
                    # Only process if this split needs processing at this chunk index
                    resume_idx_for_split = split_resume_indices.get(split, 0)
                    if chunk_idx >= resume_idx_for_split:
                        if chunk_id is None or turns is None:
                            raise ValueError(f"Item for conv_id {conv_id} must have 'chunk_id' and 'turns_json' fields.")
                        
                        # Parse turns json if stored as string
                        if isinstance(turns, str):
                            turns = json.loads(turns)
                        
                        batch_data.append({
                            'conv_id': conv_id,
                            'chunk_id': chunk_id,
                            'turns': turns,
                            'memory': conv_memories[conv_id],
                            'manager': conv_managers[conv_id],
                            'split': split,
                        })
            
            if not batch_data:
                continue  # No conversations need processing at this chunk
            
            print(f"\n=== Processing chunk index {chunk_idx} ===")
            
            # Count how many train vs val items in this batch
            if has_split_info:
                train_count = sum(1 for data in batch_data if data.get('split') == 'train')
                val_count = sum(1 for data in batch_data if data.get('split') == 'validation')
                print(f"Processing {len(batch_data)} conversations (train: {train_count}, val: {val_count})")
            else:
                print(f"Processing {len(batch_data)} conversations")
            
            # Generate and tokenize prompts
            batch_input_ids, batch_attention_mask, batch_position_ids, formatted_turns_list, formatted_memory_list = self._generate_and_tokenize_memory_prompts(batch_data)
            print(f"Created batch of size {batch_input_ids.shape[0]} for generation")
            
            # Create batched DataProto
            prompt_batch = DataProto.from_dict({
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
                'position_ids': batch_position_ids,
            })
            
            # Generate responses for entire batch
            response_batch = self._generate_with_gpu_padding(prompt_batch, actor=self.actor_rollout_wg)
            
            # Process responses and update memory states, then cache snapshots
            total_prompt_length = batch_input_ids.shape[1]

            # Execute memory operations
            _, memory_operations_per_sample = self._execute_memory_operations(batch_data, response_batch, total_prompt_length)
            
            # Cache the updated memory snapshots using split from batch_data
            for data in batch_data:
                split_to_use = data.get('split', self.split)
                data['manager'].cache_snapshot(data['memory'], data['conv_id'], data['chunk_id'], epoch, split_to_use)
                print(f"Cached snapshot for conv {data['conv_id']}, chunk {data['chunk_id']}, epoch {epoch} ({split_to_use})")
            
        print(f"\nCompleted generating memory snapshots for epoch {epoch}{split_info}")

    def _prepare_batch_data_from_gen_batch(
        self,
        gen_batch: DataProto,
        shared_manager: MemoryManager,
        conv_memories: Dict[str, Memory]
    ):
        """Prepare batch data structure from gen_batch for processing.
        
        Args:
            gen_batch: The generation batch data
            shared_manager: Shared memory manager (stateless, reusable)
            conv_memories: Dictionary of memory states keyed by "conv_id_chunk_chunkid"
            
        Returns:
            list: Batch data with conv_id, chunk_id, turns, memory, manager for each item
        """
        sample_ids = gen_batch.non_tensor_batch['sample_id']
        chunk_ids = gen_batch.non_tensor_batch['chunk_id']
        turns_list = gen_batch.non_tensor_batch['turns_json']
        
        batch_data = []
        for i in range(len(sample_ids)):
            conv_id = sample_ids[i]
            chunk_id = chunk_ids[i]
            turns = turns_list[i]
            
            # Create unique key for (conv_id, chunk_id) pair
            conv_chunk_key = f"{conv_id}_chunk_{chunk_id}"
            
            # Parse turns if stored as string
            if isinstance(turns, str):
                turns = json.loads(turns)
            
            batch_data.append({
                'conv_id': conv_id,
                'chunk_id': chunk_id,
                'turns': turns,
                'memory': conv_memories[conv_chunk_key],
                'manager': shared_manager  # Reuse the same manager for all
            })
        
        return batch_data

    def generate_memory_snapshots_with_teacher(
        self, 
        dataset, 
        epoch: int,
        teacher_model_name: str = "gemini-2.5-flash",
        teacher_cache_dir: str = None,
    ) -> None:
        """
        Generate memory snapshots using a STRONG TEACHER MODEL (Gemini API).
        
        ⭐ KEY DIFFERENCE from generate_memory_snapshots():
        - Uses FROZEN EXPERT MODEL (Gemini) instead of training policy
        - Provides HIGH-QUALITY memory states as supervision
        - No on-policy leakage (teacher is independent of training)
        - Suitable for curriculum learning and imitation learning
        
        This is the RECOMMENDED way to generate snapshots for RL training.
        The training policy will learn to imitate the expert memory operations.
        
        Args:
            dataset: The dataset containing conversations
            epoch: Current epoch number
            teacher_model_name: Gemini model to use (default: gemini-1.5-flash)
            teacher_cache_dir: Cache directory for teacher responses
        """
        print(f"\n{'='*70}")
        print(f"🎓 Generating Expert Memory Snapshots with Teacher Model")
        print(f"   Using: {teacher_model_name} (strong, frozen model)")
        print(f"   Epoch: {epoch}")
        print(f"   Split: {self.split}")
        print(f"{'='*70}\n")
        
        # Create teacher model
        teacher = TeacherModel(
            model_name=teacher_model_name,
            cache_dir=teacher_cache_dir,
            top_k_memories=self.config.top_k_memories_for_operations,
            similarity_threshold=self.config.similarity_threshold,
            temperature=0.0,  # Deterministic for reproducibility
        )
        
        # Use teacher model to generate snapshots
        teacher.generate_memory_snapshots_for_dataset(
            dataset=dataset,
            epoch=epoch,
            split=self.split,
        )
        
        print(f"\n{'='*70}")
        print(f"✅ Expert snapshots ready for RL training!")
        print(f"   Training policy will load these as supervision signals")
        print(f"{'='*70}\n")

    def _validate_combined_sequences(
        self,
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        full_response_mask: torch.Tensor,
        isolated_response_ids: torch.Tensor,
        isolated_response_attention_mask: torch.Tensor,
        original_prompts_text: list[list[str]],
        original_responses_text: list[list[str]],
    ):
        """Validate the correctness of combined sequences and masks.
        
        Args:
            full_input_ids: Combined input IDs [batch_size, total_len]
            full_attention_mask: Combined attention mask [batch_size, total_len]
            full_position_ids: Position IDs [batch_size, total_len]
            full_response_mask: Response mask [batch_size, total_len]
            isolated_response_ids: Isolated response IDs [batch_size, response_len]
            isolated_response_attention_mask: Isolated response attention mask [batch_size, response_len]
            original_prompts_text: Original prompt texts for each batch item
            original_responses_text: Original response texts for each batch item
        """
        batch_size = full_input_ids.shape[0]
        
        # 1. Check that full sequence decodes to concatenation of all original texts
        print("\n[VALIDATION] Checking decoded text consistency...")
        for batch_idx in range(min(3, batch_size)):  # Check first 3 samples
            # Decode full sequence (skip padding)
            valid_mask = full_attention_mask[batch_idx].bool()
            valid_ids = full_input_ids[batch_idx][valid_mask]
            full_decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            # Expected: concatenation of all prompts + all responses
            expected_text = ''.join(original_prompts_text[batch_idx] + original_responses_text[batch_idx])
            
            # Note: tokenizer may add/remove spaces, so we compare stripped versions
            full_decoded_stripped = full_decoded.replace(' ', '').replace('\n', '')
            expected_stripped = expected_text.replace(' ', '').replace('\n', '')
            
            assert full_decoded_stripped == expected_stripped, \
                f"Sample {batch_idx}: Full sequence text mismatch!\nDecoded: '{full_decoded}'\nExpected: '{expected_text}'"
            print(f"  ✓ Sample {batch_idx}: Full sequence text matches")
        
        # 2. Check that isolated responses decode correctly
        print("[VALIDATION] Checking isolated response consistency...")
        for batch_idx in range(min(3, batch_size)):
            # Decode isolated responses (skip padding)
            valid_mask = isolated_response_attention_mask[batch_idx].bool()
            valid_ids = isolated_response_ids[batch_idx][valid_mask]
            isolated_decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            # Expected: concatenation of all responses
            expected_text = ''.join(original_responses_text[batch_idx])
            
            isolated_decoded_stripped = isolated_decoded.replace(' ', '').replace('\n', '')
            expected_stripped = expected_text.replace(' ', '').replace('\n', '')
            
            assert isolated_decoded_stripped == expected_stripped, \
                f"Sample {batch_idx}: Isolated responses text mismatch!\nDecoded: '{isolated_decoded}'\nExpected: '{expected_text}'"
            print(f"  ✓ Sample {batch_idx}: Isolated responses text matches")
        
        # 3. Check mask coherence
        print("[VALIDATION] Checking mask coherence...")
        # full_response_mask should only be 1 in the response portion
        # Check that number of 1s in full_response_mask equals number of 1s in isolated_response_attention_mask
        for batch_idx in range(batch_size):
            full_response_tokens = full_response_mask[batch_idx].sum().item()
            isolated_response_tokens = isolated_response_attention_mask[batch_idx].sum().item()
            
            assert full_response_tokens == isolated_response_tokens, \
                f"Sample {batch_idx}: Response token count mismatch! Full: {full_response_tokens}, Isolated: {isolated_response_tokens}"
        print(f"  ✓ All {batch_size} samples: Response mask coherence verified")
        
        # 4. Check position IDs are monotonically increasing where attention_mask is 1
        print("[VALIDATION] Checking position IDs...")
        for batch_idx in range(min(3, batch_size)):
            valid_mask = full_attention_mask[batch_idx].bool()
            valid_positions = full_position_ids[batch_idx][valid_mask]
            
            # Should start from 0 and increase by 1 each step
            expected_positions = torch.arange(len(valid_positions), device=valid_positions.device)
            assert torch.equal(valid_positions, expected_positions), \
                f"Sample {batch_idx}: Position IDs are not sequential! Got: {valid_positions[:10]}"
            print(f"  ✓ Sample {batch_idx}: Position IDs are sequential (0 to {len(valid_positions)-1})")
        
        # 5. Verify format: prompts first, then responses
        print("[VALIDATION] Checking [PROMPTS|RESPONSES] format...")
        for batch_idx in range(min(3, batch_size)):
            response_mask = full_response_mask[batch_idx]
            
            # Find where responses start
            response_indices = (response_mask == 1).nonzero(as_tuple=True)[0]
            if len(response_indices) > 0:
                first_response_idx = response_indices[0].item()
                last_response_idx = response_indices[-1].item()
                
                # All responses should be contiguous at the end
                # All zeros should be before first_response_idx
                assert response_mask[:first_response_idx].sum() == 0, \
                    f"Sample {batch_idx}: Found response tokens before response section starts!"
                
                # All ones should be between first and last response index (accounting for padding)
                response_section = response_mask[first_response_idx:last_response_idx+1]
                valid_in_section = full_attention_mask[batch_idx][first_response_idx:last_response_idx+1]
                # Where attention is 1 in response section, response_mask should also be 1
                assert (response_section[valid_in_section.bool()] == 1).all(), \
                    f"Sample {batch_idx}: Response section has gaps!"
                
                print(f"  ✓ Sample {batch_idx}: Format is [PROMPTS|RESPONSES] with responses at indices [{first_response_idx}:{last_response_idx}]")
        
        print("[VALIDATION] ✓ All validation checks passed!\n")

    def combine_prompts_and_responses_for_rl_training(self, all_input_ids: list[torch.Tensor], all_attention_masks: list[torch.Tensor], is_val: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine prompts and responses into [ALL_PROMPTS | ALL_RESPONSES] format for RL training.
        
        This function takes interleaved prompts and responses and reorganizes them into a format
        suitable for multiturn RL training where all prompts are concatenated first, followed by
        all responses. This enables proper loss masking during training.
        
        Args:
            all_input_ids: List of input_id tensors, each of shape [batch_size, seq_len].
                          Order: [memory_prompt, memory_response, qa_prompt_1, qa_response_1, ...]
            all_attention_masks: List of attention_mask tensors, each of shape [batch_size, seq_len].
        
        Returns:
            tuple: (full_input_ids, full_attention_mask, full_position_ids, full_response_mask, 
                    isolated_response_ids, isolated_response_attention_mask, isolated_prompt_ids)
                   - full_input_ids: [batch_size, max_prompts_len + max_responses_len] - Full sequence
                   - full_attention_mask: [batch_size, max_prompts_len + max_responses_len] - 1 for valid tokens
                   - full_position_ids: [batch_size, max_prompts_len + max_responses_len] - Position IDs
                   - full_response_mask: [batch_size, max_prompts_len + max_responses_len] - 1 for response tokens only
                   - isolated_response_ids: [batch_size, max_responses_len] - Isolated responses (right-padded)
                   - isolated_response_attention_mask: [batch_size, max_responses_len] - Mask for isolated responses
                   - isolated_prompt_ids: [batch_size, max_prompts_len] - Isolated prompts (left-padded)
        """
        # Note that we have
        #  all_input_ids        = [memory_prompt, memory_response, qa_prompt_1, qa_response_1, qa_prompt_2, qa_response_2, ...]
        #  all_attention_masks  = [memory_prompt, memory_response, qa_prompt_1, qa_response_1, qa_prompt_2, qa_response_2, ...]
        
        # We need to rearrange into: [memory_prompt, qa_prompt_1, qa_prompt_2, ..., memory_response, qa_response_1, qa_response_2, ...]
        # This is the expected format for multiturn RL training: all prompts first, then all responses
        
        # Strategy:
        # 1. Separate prompts and responses, remove their padding because we cant have paddings in middle of sequence
        # 2. Pad all prompts combined to max_prompts_len (LEFT padding)
        # 3. Pad all responses combined to max_responses_len (RIGHT padding)
            # So in the end we have [PAD, ALL_PROMPTS] and [ALL_RESPONSES, PAD]
        # 4. Concatenate: [padded_prompts, padded_responses]
        
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = all_input_ids[0].shape[0]
        
        # Store original decoded texts for validation
        original_prompts_text = [[] for _ in range(batch_size)]
        original_responses_text = [[] for _ in range(batch_size)]
        
        # Process each item in the batch separately to collect prompts and responses
        batch_all_prompts_ids = []
        batch_all_prompts_masks = []
        batch_all_responses_ids = []
        batch_all_responses_masks = []
        
        for batch_idx in range(batch_size):
            # Separate prompts and responses
            prompt_segments_ids = [] # in end, should be of len = num_prompts
            prompt_segments_masks = [] # in end, should be of len = num_prompts
            response_segments_ids = [] # in end, should be of len = num_responses
            response_segments_masks = [] # in end, should be of len = num_responses
            
            for segment_idx in range(len(all_input_ids)):
                input_ids = all_input_ids[segment_idx][batch_idx]  # [seq_len]
                attention_mask = all_attention_masks[segment_idx][batch_idx]  # [seq_len]
                
                # Store original text for validation
                original_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                
                # For prompts (even indices: 0, 2, 4, ...), padding is on the LEFT
                # For responses (odd indices: 1, 3, 5, ...), padding is on the RIGHT
                is_prompt = (segment_idx % 2 == 0)
                
                if is_prompt:
                    # Remove LEFT padding (keep only non-padded tokens)
                    # Find first non-pad token
                    non_pad_mask = attention_mask.bool()
                    if not non_pad_mask.any():
                        # Empty prompt segment (all padding), so skip !!
                        # print(f"Skipping empty prompt segment {segment_idx} in batch {batch_idx}")
                        continue

                    first_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][0]
                    input_ids = input_ids[first_non_pad:]
                    attention_mask = attention_mask[first_non_pad:]

                    # Validation: decoded text should match after padding removal
                    unpadded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    assert original_text == unpadded_text, \
                        f"Prompt text mismatch after padding removal! Original: '{original_text}' vs Unpadded: '{unpadded_text}'"
                    
                    prompt_segments_ids.append(input_ids)
                    prompt_segments_masks.append(attention_mask)
                    original_prompts_text[batch_idx].append(original_text)
                else:
                    # Validation: response segment length should match configured max_response_length_per_turn
                    # I made it smaller or equal because i can give more space to fill memory
                    assert input_ids.shape[0] <= self.config.max_response_length_per_turn, \
                        (f"Response segment length {input_ids.shape[0]} does not match configured max_response_length_per_turn {self.config.max_response_length_per_turn}!")
                    
                    # Remove RIGHT padding (keep only non-padded tokens)
                    # Find last non-pad token
                    non_pad_mask = attention_mask.bool()
                    if not non_pad_mask.any():
                        # Empty response segment (all padding), so skip !!
                        # print(f"Skipping empty response segment {segment_idx} in batch {batch_idx}")
                        continue
                    
                    last_non_pad = non_pad_mask.nonzero(as_tuple=True)[0][-1]
                    input_ids = input_ids[:last_non_pad + 1]
                    attention_mask = attention_mask[:last_non_pad + 1]
                    
                    # Validation: decoded text should match after padding removal
                    unpadded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    assert original_text == unpadded_text, \
                        f"Response text mismatch after padding removal! Original: '{original_text}' vs Unpadded: '{unpadded_text}'"
                    
                    response_segments_ids.append(input_ids)
                    response_segments_masks.append(attention_mask)
                    original_responses_text[batch_idx].append(original_text)
            
            # Concatenate ALL prompts and ALL responses for this batch item
            all_prompts_ids = torch.cat(prompt_segments_ids, dim=0)
            all_prompts_masks = torch.cat(prompt_segments_masks, dim=0)
            all_responses_ids = torch.cat(response_segments_ids, dim=0)
            all_responses_masks = torch.cat(response_segments_masks, dim=0)
            
            batch_all_prompts_ids.append(all_prompts_ids)
            batch_all_prompts_masks.append(all_prompts_masks)
            batch_all_responses_ids.append(all_responses_ids)
            batch_all_responses_masks.append(all_responses_masks)
        
        # Find maximum prompt length and maximum response length across the batch
        max_prompts_len = max(seq.shape[0] for seq in batch_all_prompts_ids)
        max_responses_len = max(seq.shape[0] for seq in batch_all_responses_ids)

        if not is_val:
            # For validation, we can be flexible and allow longer sequences by truncating later
            # since we are not training on them, but for training we must enforce limits
            assert max_prompts_len <= self.config.max_prompt_length, \
                f"Max prompts length {max_prompts_len} exceeds configured max_prompt_length {self.config.max_prompt_length}!"
            
            assert max_responses_len <= self.config.max_response_length, \
                f"Max responses length {max_responses_len} exceeds configured max_response_length {self.config.max_response_length}!"
        
        # Pad prompts to max_prompts_len (LEFT padding) and responses to max_responses_len (RIGHT padding)
        padded_prompts_ids = []
        padded_prompts_masks = []
        padded_responses_ids = []
        padded_responses_masks = []
        
        for prompts_ids, prompts_masks, responses_ids, responses_masks in zip(
            batch_all_prompts_ids, batch_all_prompts_masks, 
            batch_all_responses_ids, batch_all_responses_masks
        ):
            # Pad prompts with LEFT padding
            prompts_len = prompts_ids.shape[0]
            prompts_pad_len = max_prompts_len - prompts_len
            if prompts_pad_len > 0:
                left_pad_ids = torch.full((prompts_pad_len,), pad_token_id, dtype=prompts_ids.dtype, device=prompts_ids.device)
                left_pad_mask = torch.zeros((prompts_pad_len,), dtype=prompts_masks.dtype, device=prompts_masks.device)
                prompts_ids = torch.cat([left_pad_ids, prompts_ids], dim=0)
                prompts_masks = torch.cat([left_pad_mask, prompts_masks], dim=0)
            
            # Pad responses with RIGHT padding
            responses_len = responses_ids.shape[0]
            responses_pad_len = max_responses_len - responses_len
            if responses_pad_len > 0:
                right_pad_ids = torch.full((responses_pad_len,), pad_token_id, dtype=responses_ids.dtype, device=responses_ids.device)
                right_pad_mask = torch.zeros((responses_pad_len,), dtype=responses_masks.dtype, device=responses_masks.device)
                responses_ids = torch.cat([responses_ids, right_pad_ids], dim=0)
                responses_masks = torch.cat([responses_masks, right_pad_mask], dim=0)
            
            padded_prompts_ids.append(prompts_ids)
            padded_prompts_masks.append(prompts_masks)
            padded_responses_ids.append(responses_ids)
            padded_responses_masks.append(responses_masks)
        
        # Now concatenate prompts and responses for each batch item
        full_input_ids_list = []
        full_attention_mask_list = []
        full_response_mask_list = []
        
        for prompts_ids, prompts_masks, responses_ids, responses_masks in zip(
            padded_prompts_ids, padded_prompts_masks,
            padded_responses_ids, padded_responses_masks
        ):
            # Concatenate: [prompts, responses]
            combined_ids = torch.cat([prompts_ids, responses_ids], dim=0)
            combined_mask = torch.cat([prompts_masks, responses_masks], dim=0)
            
            # Create response mask: 0 for prompts and prompt padding, 1 ONLY for actual response tokens
            # Use responses_masks (attention mask) to ensure we only mark valid response tokens
            prompt_response_mask = torch.zeros_like(prompts_masks)
            response_response_mask = responses_masks  # This already has 0s for padding, 1s for valid tokens
            combined_response_mask = torch.cat([prompt_response_mask, response_response_mask], dim=0)
            
            full_input_ids_list.append(combined_ids)
            full_attention_mask_list.append(combined_mask)
            full_response_mask_list.append(combined_response_mask)
        
        # Stack into batch tensors [batch_size, max_prompts_len + max_responses_len]
        full_input_ids = torch.stack(full_input_ids_list, dim=0)
        full_attention_mask = torch.stack(full_attention_mask_list, dim=0)
        full_position_ids = compute_position_id_with_mask(full_attention_mask)
        full_response_mask = torch.stack(full_response_mask_list, dim=0)
        
        # Also stack the padded responses separately for easy access
        isolated_response_ids = torch.stack(padded_responses_ids, dim=0)
        isolated_response_attention_mask = torch.stack(padded_responses_masks, dim=0)
        
        # Also stack the padded prompts separately for easy access
        isolated_prompt_ids = torch.stack(padded_prompts_ids, dim=0)
        
        # Run validation checks
        # self._validate_combined_sequences(
        #     full_input_ids,
        #     full_attention_mask,
        #     full_position_ids,
        #     full_response_mask,
        #     isolated_response_ids,
        #     isolated_response_attention_mask,
        #     original_prompts_text,
        #     original_responses_text,
        # )
        
        # print(f"[combine_prompts_and_responses_for_rl_training] Full sequence shape: {full_input_ids.shape}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Isolated response IDs shape: {isolated_response_ids.shape}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Isolated response attention mask shape: {isolated_response_attention_mask.shape}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Isolated prompt IDs shape: {isolated_prompt_ids.shape}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Batch size: {batch_size}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Max prompts length: {max_prompts_len}, Max responses length: {max_responses_len}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Total combined length: {max_prompts_len + max_responses_len}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Format: [PROMPTS (left-padded), RESPONSES (right-padded)]")
        # print(f"[combine_prompts_and_responses_for_rl_training] Response tokens in full sequence per sample (first 5): {full_response_mask.sum(dim=1)[:5].tolist()}")
        # print(f"[combine_prompts_and_responses_for_rl_training] Response tokens in isolated per sample (first 5): {isolated_response_attention_mask.sum(dim=1)[:5].tolist()}")
        
        return full_input_ids, full_attention_mask, full_position_ids, full_response_mask, isolated_response_ids, isolated_response_attention_mask, isolated_prompt_ids

    def run_chunks_validation(self, gen_batch: DataProto, global_step: int) -> DataProto:
        """Run validation with pre-generated memory snapshots (LOAD-ONLY mode).
        
        This is a simplified version of run_chunks() specifically for validation:
        - NO sequential dependency (batch can have mixed chunk_ids)
        - NO memory state caching (read-only from pre-generated snapshots)
        - NO policy updates (frozen model)
        
        This allows efficient batching of validation samples regardless of chunk_id.
        
        Args:
            gen_batch: The generation batch data (can have mixed chunk_ids).
            global_step: The global training step at which validation is running.
            
        Returns:
            DataProto with prompts, responses, and rewards for validation evaluation.
        """
        print("Running validation memory generation (load-only mode)...")
        non_tensor_batch = gen_batch.non_tensor_batch if hasattr(gen_batch, 'non_tensor_batch') else None

        sample_ids = non_tensor_batch['sample_id']
        chunk_ids = non_tensor_batch['chunk_id']
        speakers = non_tensor_batch['speakers']
        qa_pairs = non_tensor_batch['qa_pairs_json']
        
        print(f"Processing validation batch with {len(sample_ids)} samples (chunk_ids: {set(chunk_ids)})")
        
        # Parse qa_pairs once
        for i in range(len(qa_pairs)):
            qa_pairs[i] = json.loads(qa_pairs[i]) if isinstance(qa_pairs[i], str) else qa_pairs[i]
        
        # Track variable QA counts per sample (validation can also have variable counts)
        qa_counts = [len(qa_pairs[i]) for i in range(len(qa_pairs))]
        total_qa_count = sum(qa_counts)
        print(f"[VALIDATION] QA counts per sample: {qa_counts} (total={total_qa_count} QAs)")
        
        # Initialize shared memory manager
        shared_manager = MemoryManager()
        
        # Load pre-generated memory snapshots for each (conv_id, chunk_id) pair
        # NO sequential dependency - each loads its own pre-generated snapshot directly
        conv_memories: Dict[str, Memory] = {}
        batch_data = []
        
        for i in range(len(sample_ids)):
            conv_id = sample_ids[i]
            chunk_id = chunk_ids[i]
            turns = non_tensor_batch['turns_json'][i]
            
            if isinstance(turns, str):
                turns = json.loads(turns)
            
            # Load PRE-GENERATED snapshot for this specific (conv_id, chunk_id)
            # This was generated by generate_memory_snapshots() at validation start
            if chunk_id > 1:
                prev_chunk_id = chunk_id - 1
                loaded_memory = shared_manager.get_snapshot(conv_id, prev_chunk_id, global_step, self.split)
                if loaded_memory is None:
                    raise Exception(f"[VALIDATION] No cached memory for conv {conv_id}, chunk {prev_chunk_id}. "
                                  f"Ensure generate_memory_snapshots() ran successfully.")
                conv_chunk_key = f"{conv_id}_chunk_{chunk_id}"
                conv_memories[conv_chunk_key] = loaded_memory
                print(f"[VALIDATION] Loaded memory for conv {conv_id} from chunk {prev_chunk_id}")
            else:
                # First chunk, start with empty memory
                conv_chunk_key = f"{conv_id}_chunk_{chunk_id}"
                conv_memories[conv_chunk_key] = Memory()
            
            batch_data.append({
                'conv_id': conv_id,
                'chunk_id': chunk_id,
                'turns': turns,
                'memory': conv_memories[conv_chunk_key],
                'manager': shared_manager
            })
        
        # Rest of the logic is identical to run_chunks() but WITHOUT caching
        all_inputs_ids = []
        all_attention_masks = []
        
        # Generate memory operations
        batch_input_ids, batch_attention_mask, batch_position_ids, formatted_turns, formatted_memory = self._generate_and_tokenize_memory_prompts(batch_data)
        all_inputs_ids.append(batch_input_ids)
        all_attention_masks.append(batch_attention_mask)
        
        memory_prompt_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids,
        })

        # Must copy meta_info to reserve that we are in validation mode
        memory_prompt_batch.meta_info = gen_batch.meta_info
        
        memory_response_batch = self._generate_with_gpu_padding(memory_prompt_batch, actor=self.actor_rollout_wg)
        total_prompt_length = batch_input_ids.shape[1]
        
        all_inputs_ids.append(memory_response_batch.batch['input_ids'][:, total_prompt_length:])
        all_attention_masks.append(memory_response_batch.batch['attention_mask'][:, total_prompt_length:])
        
        # Execute memory operations (updates memory in-place but we DON'T cache)
        format_rewards, memory_operations_per_sample = self._execute_memory_operations(batch_data, memory_response_batch, total_prompt_length)

        # Evaluate memory operations
        memory_rewards = self._evaluate_memory_operations(formatted_turns, formatted_memory, memory_operations_per_sample)
        
        assert total_qa_count > 0, "Total QA count after sampling must be greater than 0"
        # ========================================
        # BATCHED QA GENERATION OPTIMIZATION (VARIABLE LENGTH)
        # ========================================
        # Instead of looping over questions sequentially, we batch ALL questions at once.
        # Note: With variable QA counts, we process all available QAs across all samples.
        
        full_answer_rewards = [[] for _ in range(len(batch_data))]
        
        print(f"[VALIDATION] Generating answers for ALL {total_qa_count} questions in one batch")
        
        # Step 1: Create ALL QA prompts at once (sum of all QAs across samples)
        all_qa_prompts = []
        qa_metadata = []  # Track (sample_idx, question_idx_within_sample) for each prompt
        
        # Iterate over each sample and create prompts for all its QAs
        for i, data in enumerate(batch_data):
            num_qas_for_sample = len(qa_pairs[i])
            
            for question_idx in range(num_qas_for_sample):
                qa_prompt = generate_qa_prompt(
                    data['memory'],
                    speaker_1=speakers[i][0],
                    speaker_2=speakers[i][1],
                    question=qa_pairs[i][question_idx]['question'],
                    top_k_per_speaker=self.config.top_k_memories_for_qa // 2,
                    similarity_threshold=self.config.similarity_threshold,
                    use_similarity=True
                )
                all_qa_prompts.append(qa_prompt)
                qa_metadata.append((i, question_idx))  # (sample_idx, question_idx_within_sample)
        
        # Step 2: Tokenize ALL prompts at once
        qa_input_ids, qa_attention_masks, qa_position_ids = self._generate_and_tokenize_prompts(all_qa_prompts)
        
        # Step 3: Generate ALL responses in ONE batched call
        qa_prompt_batch = DataProto.from_dict({
            'input_ids': qa_input_ids,
            'attention_mask': qa_attention_masks,
            'position_ids': qa_position_ids,
        })
        
        # Must copy meta_info to reserve that we are in validation mode
        qa_prompt_batch.meta_info = gen_batch.meta_info
        qa_response_batch = self._generate_with_gpu_padding(qa_prompt_batch, actor=self.actor_rollout_wg)
        total_prompt_length_of_qa = qa_input_ids.shape[1]
        
        print(f"[VALIDATION] Generated {len(all_qa_prompts)} QA responses in one batch (total across all samples with variable counts: {qa_counts})")
        
        # Step 4: Process responses and evaluate answers
        judge_evaluation_batch = []
        for flat_idx, (sample_idx, question_idx) in enumerate(qa_metadata):
            response_ids = qa_response_batch.batch['input_ids'][flat_idx, total_prompt_length_of_qa:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            predicted_answer = extract_answer_from_text(response_text)
            if predicted_answer is not None:
                predicted_answer = str(predicted_answer).strip()
            else:
                predicted_answer = response_text.strip()
            
            gold_answer = str(qa_pairs[sample_idx][question_idx]['answer']).strip()
            question = qa_pairs[sample_idx][question_idx]['question']
            
            gold_answer = gold_answer.strip('\'"')
            predicted_answer = predicted_answer.strip('\'"')
            
            judge_evaluation_batch.append({
                'sample_idx': sample_idx,
                'question_idx': question_idx,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'conv_id': batch_data[sample_idx]['conv_id'],
                'chunk_id': batch_data[sample_idx]['chunk_id']
            })
        
        # Step 5: Evaluate ALL answers at once
        print(f"[VALIDATION] Evaluating {len(judge_evaluation_batch)} answers with {'LLM judge' if self.config.use_llm_judge else 'string matching'}")
        all_judge_scores = self._evaluate_answers_with_judge_batched(
            judge_evaluation_batch, 
            use_llm_judge=self.config.use_llm_judge
        )
        
        # Step 6: Organize rewards back to [batch_size][variable_qa_count] structure
        for eval_item, judge_score in zip(judge_evaluation_batch, all_judge_scores):
            sample_idx = eval_item['sample_idx']
            full_answer_rewards[sample_idx].append(judge_score)
        
        # Step 7: Add to all_inputs for later concatenation (handle variable QA counts)
        # Group QA results back by sample (similar to training code)
        sample_qa_prompts = [[] for _ in range(len(batch_data))]
        sample_qa_prompt_masks = [[] for _ in range(len(batch_data))]
        sample_qa_responses = [[] for _ in range(len(batch_data))]
        sample_qa_response_masks = [[] for _ in range(len(batch_data))]
        
        for flat_idx, (sample_idx, question_idx) in enumerate(qa_metadata):
            sample_qa_prompts[sample_idx].append(qa_input_ids[flat_idx])
            sample_qa_prompt_masks[sample_idx].append(qa_attention_masks[flat_idx])
            sample_qa_responses[sample_idx].append(qa_response_batch.batch['input_ids'][flat_idx, total_prompt_length_of_qa:])
            sample_qa_response_masks[sample_idx].append(qa_response_batch.batch['attention_mask'][flat_idx, total_prompt_length_of_qa:])
        
        # Now add QAs in interleaved format for each sample, padding to max_qa_count
        max_qa_count = max(qa_counts)
        
        # Find a sample with at least one QA to use as template for empty tensors
        template_sample_idx = None
        for idx, count in enumerate(qa_counts):
            if count > 0:
                template_sample_idx = idx
                break
        
        if template_sample_idx is None:
            raise ValueError("[VALIDATION] All samples have 0 QAs - cannot create batch structure")
        
        # Add QA pairs question by question (to maintain interleaved structure)
        for qa_idx in range(max_qa_count):
            qa_prompts_batch = []
            qa_prompt_masks_batch = []
            qa_responses_batch = []
            qa_response_masks_batch = []
            
            for sample_idx in range(len(batch_data)):
                if qa_idx < qa_counts[sample_idx]:
                    # Sample has this QA
                    qa_prompts_batch.append(sample_qa_prompts[sample_idx][qa_idx])
                    qa_prompt_masks_batch.append(sample_qa_prompt_masks[sample_idx][qa_idx])
                    qa_responses_batch.append(sample_qa_responses[sample_idx][qa_idx])
                    qa_response_masks_batch.append(sample_qa_response_masks[sample_idx][qa_idx])
                else:
                    # Sample doesn't have this QA - create empty tensors with same shape as template
                    empty_prompt = torch.full_like(sample_qa_prompts[template_sample_idx][0], self.tokenizer.pad_token_id)
                    empty_mask = torch.zeros_like(sample_qa_prompt_masks[template_sample_idx][0])
                    qa_prompts_batch.append(empty_prompt)
                    qa_prompt_masks_batch.append(empty_mask)
                    
                    empty_response = torch.full_like(sample_qa_responses[template_sample_idx][0], self.tokenizer.pad_token_id)
                    empty_response_mask = torch.zeros_like(sample_qa_response_masks[template_sample_idx][0])
                    qa_responses_batch.append(empty_response)
                    qa_response_masks_batch.append(empty_response_mask)
            
            all_inputs_ids.append(torch.stack(qa_prompts_batch))
            all_attention_masks.append(torch.stack(qa_prompt_masks_batch))
            all_inputs_ids.append(torch.stack(qa_responses_batch))
            all_attention_masks.append(torch.stack(qa_response_masks_batch))
        
        # Compute total rewards (no GRPO selection, no caching)
        qa_rewards_averaged = [np.mean(rewards) if rewards else 0.0 for rewards in full_answer_rewards]
        total_rewards = [
            self.config.format_reward_weight * format_rewards[i] + 
            self.config.answer_reward_weight * qa_rewards_averaged[i] +
            self.config.memory_reward_weight * memory_rewards[i]
            for i in range(len(batch_data))
        ]
        
        print(f"[VALIDATION] Completed batch - NOT caching memory states (validation is read-only)")
        
        # Combine prompts and responses
        full_input_ids, full_attention_mask, full_position_ids, full_response_mask, \
            isolated_response_ids, isolated_response_attention_mask, isolated_prompt_ids = \
            self.combine_prompts_and_responses_for_rl_training(all_inputs_ids, all_attention_masks, is_val=True)

        # Normalize rewards if needed across the batch
        epsilon = 1e-8
        total_rewards = np.array(total_rewards)
        mu = np.mean(total_rewards)
        sigma = np.std(total_rewards) + epsilon
        total_rewards_normalized = (total_rewards - mu) / sigma
        
        result = DataProto.from_dict({
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'position_ids': full_position_ids,
            'responses': isolated_response_ids,
            'response_mask': isolated_response_attention_mask,
            'prompts': isolated_prompt_ids,
        }, meta_info=gen_batch.meta_info)
        
        result.non_tensor_batch = {
            'format_rewards': np.array(format_rewards, dtype=object),
            'full_answer_rewards': np.array(full_answer_rewards, dtype=object),
            'qa_rewards_averaged': np.array(qa_rewards_averaged, dtype=object),
            'memory_rewards': np.array(memory_rewards, dtype=object),
            'total_rewards_unnormalized': np.array(total_rewards, dtype=object),  # Array of floats (weighted total)
            'total_rewards': np.array(total_rewards_normalized, dtype=object),
        }
        
        return result

    def run_chunks(self, gen_batch: DataProto, epoch: int) -> DataProto:
        """Run the memory loop for generation with memory operations (ON-POLICY).
        
        CRITICAL: This function generates memory states ON-THE-FLY during training with the CURRENT policy.
        Memory states are generated sequentially within each epoch and cached for next chunk in same epoch.
        
        Args:
            gen_batch: The generation batch data.
            epoch: The current epoch number.

        Flow:
        1. Load memory state from previous chunk (generated earlier in THIS epoch with current policy)
           - For chunk 1: Start with empty memory
           - For chunk i>1: Load state from chunk i-1 (generated by current policy in this epoch)
        
        2. Generate memory operations with CURRENT policy:
           a. Create memory prompt using current memory state and chunk turns
           b. Generate memory operations (insert/update/delete) with current policy
           c. Execute operations to update memory state IN-PLACE
           d. DEFER caching until after all rewards are computed (for GRPO best-rollout selection)
        
        3. Generate QA responses with CURRENT policy using updated memory:
           a. For each question, retrieve relevant memories
           b. Generate answer using current policy
           c. Evaluate answer quality
        
        4. Cache best rollout's memory state (GRPO-aware):
           - If GRPO: Cache only the rollout with highest total reward (format + QA)
           - If single rollout: Cache that rollout's memory state
           - Next chunk will load this cached state
        
        5. Return combined prompts+responses for RL training:
           - Rewards based on: (1) memory operation format, (2) QA correctness
           - Policy gradient updates based on these rewards
           - ALL rollouts contribute to learning, but only BEST rollout's memory is cached
        
        IMPORTANT: 
        - Memory states are cached per-epoch, so each epoch generates fresh memory states
        - With GRPO (multiple rollouts per chunk), only the best rollout's memory is cached
        - This provides greedy exploitation while GRPO provides exploration through learning
        """
        # Extract data from batch
        print("Running memory generation for batch...")
        # print("gen_batch", gen_batch)
        non_tensor_batch = gen_batch.non_tensor_batch if hasattr(gen_batch, 'non_tensor_batch') else None

        sample_ids = non_tensor_batch['sample_id']
        chunk_ids = non_tensor_batch['chunk_id']
        
        # CRITICAL VALIDATION: All samples in batch must have same chunk_id
        # This is required for sequential memory operations (chunk N needs chunk N-1's memory)
        # unique_chunks = set(chunk_ids)
        # if len(unique_chunks) > 1:
        #     raise ValueError(
        #         f"Batch contains mixed chunk_ids: {unique_chunks}. "
        #         f"All samples must have the same chunk_id for sequential memory operations. "
        #         f"Use ChunkSequentialSampler and set use_chunk_sequential_sampler=true in config."
        #     )
        
        # current_chunk_id = chunk_ids[0]
        # print(f"Processing chunk_id={current_chunk_id} with {len(sample_ids)} conversations")
        speakers = non_tensor_batch['speakers']
        qa_pairs = non_tensor_batch['qa_pairs_json']
        num_questions = non_tensor_batch['num_qas'][0] # Since all have same number of questions per chunk

        # Saving the prompts and responses for rewards
        all_inputs_ids = []
        all_attention_masks = []
        
        # Initialize ONE shared memory manager (it's stateless, can be reused!)
        shared_manager = MemoryManager()
        
        # Only need to track memories, not managers
        conv_memories: Dict[str, Memory] = {}
        
        # Load memory states for each conversation-chunk pair from previous chunk (chunk_id - 1)
        for i in range(len(sample_ids)):
            conv_id = sample_ids[i]
            chunk_id = chunk_ids[i]
            
            # Create unique key for (conv_id, chunk_id) pair - THIS IS IMPORTANT!
            # conv_id alone is NOT unique (same conv can appear with different chunks)
            conv_chunk_key = f"{conv_id}_chunk_{chunk_id}"
            
            # Load memory from previous chunk if it exists
            if chunk_id > 1:  # chunk_id starts from 1
                prev_chunk_id = chunk_id - 1
                # Load the memory snapshot that the previous chunk_id had saved as our starting memory state
                loaded_memory = shared_manager.get_snapshot(conv_id, prev_chunk_id, epoch, self.split)
                if loaded_memory is not None:
                    conv_memories[conv_chunk_key] = loaded_memory
                    print(f"Loaded memory for conv {conv_id} from chunk {prev_chunk_id}")
                else:
                    raise Exception(f"No cached memory for conv {conv_id}, chunk {prev_chunk_id}. Required for processing.")
            else:
                # First chunk, start with empty memory
                # print(f"Starting fresh memory for conv {conv_id}, chunk {chunk_id}")
                conv_memories[conv_chunk_key] = Memory()
        
        # Prepare batch data structure
        batch_data = self._prepare_batch_data_from_gen_batch(gen_batch, shared_manager, conv_memories)
        
        # Validate batch is not empty
        if not batch_data:
            raise ValueError("Batch data is empty after preparation. Check gen_batch contents.")
        
        # Generate and tokenize prompts (reuse helper method)
        batch_input_ids, batch_attention_mask, batch_position_ids, formatted_turns, formatted_memory = self._generate_and_tokenize_memory_prompts(batch_data)

        # Append to all inputs as we need them for reward calculation later
        all_inputs_ids.append(batch_input_ids)
        all_attention_masks.append(batch_attention_mask)

        # Create batched DataProto for memory operations
        memory_prompt_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids,
        })
        
        # Must copy meta_info to reserve that we are in training mode
        memory_prompt_batch.meta_info = gen_batch.meta_info

        # Generate memory operations responses
        memory_response_batch = self._generate_with_gpu_padding(memory_prompt_batch, actor=self.actor_rollout_wg)
        # print(f"Generated memory operation responses for batch of size {batch_input_ids.shape[0]}")
        # print("memory_prompt_batch.shape =", memory_prompt_batch.batch['input_ids'].shape)
        # print("memory_response_batch.shape =", memory_response_batch.batch['input_ids'].shape)
        # print("memory_response_batch from total prompt length:", memory_response_batch.batch['input_ids'][:, batch_input_ids.shape[1]:])
        # print("memory_response_batch attention from total prompt length:", memory_response_batch.batch['attention_mask'][:, batch_input_ids.shape[1]:])

        # Process memory operations (reuse helper method, but without caching)
        total_prompt_length = batch_input_ids.shape[1]
        # Get maximum effective prompt length of batch input
        max_effective_prompt_length = batch_position_ids[:, -1].max().item() + 1
        # print("max_effective_prompt_length =", max_effective_prompt_length)
        # print("total_prompt_length =", total_prompt_length)

        # Add memory_response_batch info to all responses for reward calculation later
        # Note we only need the generated part after prompt
        all_inputs_ids.append(memory_response_batch.batch['input_ids'][:, total_prompt_length:])
        all_attention_masks.append(memory_response_batch.batch['attention_mask'][:, total_prompt_length:])        

        # Execute memory operations and update memory states IN-PLACE
        # IMPORTANT: These operations affect the memory state that will be used for:
        #   1. QA generation in this batch (immediate effect)
        #   2. Next chunk in this conversation during THIS epoch (sequential effect)
        format_rewards, memory_operations_per_sample = self._execute_memory_operations(batch_data, memory_response_batch, total_prompt_length)
        
        # Evaluate memory operations
        memory_rewards = self._evaluate_memory_operations(formatted_turns, formatted_memory, memory_operations_per_sample)

        # DON'T CACHE YET - with GRPO we need to wait for full rewards (format + QA)
        # to select the best rollout before caching
        # Caching will happen after computing total rewards at the end of this function
        
        assert len(format_rewards) == len(batch_data), "Rewards length must match batch size"
        assert len(memory_rewards) == len(batch_data), "Memory rewards length must match batch size"

        # Note that full_answer_rewards should be of size batch_size and each element is array of rewards for each question
        # so if we have batch of 64 and each chunk ahas 6 questions, full_answer_rewards will be list of 64 elements each is list of 6 rewards
        full_answer_rewards = [[] for _ in range(len(batch_data))]
        
        # Parse qa_pairs once before the question loop (avoid repeated parsing)
        for i in range(len(qa_pairs)):
            qa_pairs[i] = json.loads(qa_pairs[i]) if isinstance(qa_pairs[i], str) else qa_pairs[i]

        # ========================================
        # SAMPLE QAs PER CHUNK (VARIABLE LENGTH)
        # ========================================
        # Instead of padding with future/past questions, we sample up to max_qas_per_chunk
        # from the available current questions. This creates variable-length training batches
        # which is better for on-policy RL (consistent difficulty across training).
        
        sampled_qa_pairs = []
        sampled_qa_counts = []  # Track how many QAs each sample has
        
        for i in range(len(qa_pairs)):
            available_qas = qa_pairs[i]
            num_available = len(available_qas)
            
            if num_available <= self.config.max_qas_per_chunk:
                # Use all available QAs
                sampled_qa_pairs.append(available_qas)
                sampled_qa_counts.append(num_available)
            else:
                # Randomly sample max_qas_per_chunk questions
                import random
                sampled_indices = random.sample(range(num_available), self.config.max_qas_per_chunk)
                sampled_qas = [available_qas[idx] for idx in sorted(sampled_indices)]
                sampled_qa_pairs.append(sampled_qas)
                sampled_qa_counts.append(self.config.max_qas_per_chunk)
        
        # Replace qa_pairs with sampled version
        qa_pairs = sampled_qa_pairs
        
        # Calculate total number of QAs across all samples (sum of variable counts)
        total_qa_count = sum(sampled_qa_counts)
        
        print(f"\n--- Sampled QAs per chunk (max={self.config.max_qas_per_chunk}) ---")
        print(f"QA counts per sample: {sampled_qa_counts} (total={total_qa_count} QAs across {len(batch_data)} samples)")
        
        assert total_qa_count > 0, "Total QA count after sampling must be greater than 0"
        # ========================================
        # BATCHED QA GENERATION OPTIMIZATION
        # ========================================
        # Instead of looping over questions sequentially, we batch ALL questions at once.
        # Note: With variable QA counts, we process all available QAs across all samples.
        print(f"--- Generating answers for {total_qa_count} questions in one batched call ---")
        
        # Step 1: Create ALL QA prompts at once (sum of sampled QAs across all samples)
        all_qa_prompts = []
        qa_metadata = []  # Track (sample_idx, question_idx_within_sample) for each prompt
        
        # Iterate over each sample and create prompts for its sampled QAs
        for i, data in enumerate(batch_data):
            num_qas_for_sample = len(qa_pairs[i])
            
            for question_idx in range(num_qas_for_sample):
                qa_prompt = generate_qa_prompt(
                    data['memory'],
                    speaker_1=speakers[i][0],
                    speaker_2=speakers[i][1],
                    question=qa_pairs[i][question_idx]['question'],
                    top_k_per_speaker=self.config.top_k_memories_for_qa // 2,
                    similarity_threshold=self.config.similarity_threshold,
                    use_similarity=True
                )
                all_qa_prompts.append(qa_prompt)
                qa_metadata.append((i, question_idx))  # (sample_idx, question_idx_within_sample)
        
        print(f"Created {len(all_qa_prompts)} QA prompts (total across all samples with variable counts: {sampled_qa_counts})")
        
        # Step 2: Tokenize ALL prompts at once
        qa_input_ids, qa_attention_masks, qa_position_ids = self._generate_and_tokenize_prompts(all_qa_prompts)
        
        # Step 3: Generate ALL responses in ONE batched call
        qa_prompt_batch = DataProto.from_dict({
            'input_ids': qa_input_ids,
            'attention_mask': qa_attention_masks,
            'position_ids': qa_position_ids,
        })
        
        # Must copy meta_info to reserve that we are in training mode
        qa_prompt_batch.meta_info = gen_batch.meta_info

        qa_response_batch = self._generate_with_gpu_padding(qa_prompt_batch, actor=self.actor_rollout_wg)
        total_prompt_length_of_qa = qa_input_ids.shape[1]
        
        print(f"Generated {len(all_qa_prompts)} QA responses in one batch")
        
        # Step 4: Process responses and collect evaluation data
        judge_evaluation_batch = []
        for flat_idx, (sample_idx, question_idx) in enumerate(qa_metadata):
            response_ids = qa_response_batch.batch['input_ids'][flat_idx, total_prompt_length_of_qa:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            predicted_answer = extract_answer_from_text(response_text)
            if predicted_answer is not None:
                predicted_answer = str(predicted_answer).strip()
            else:
                predicted_answer = response_text.strip()  # Use full response if extraction fails
            
            gold_answer = str(qa_pairs[sample_idx][question_idx]['answer']).strip()
            question = qa_pairs[sample_idx][question_idx]['question']
            
            # Clean up quotes if present
            gold_answer = gold_answer.strip('\'"')
            predicted_answer = predicted_answer.strip('\'"')
            
            # Store for batched judge evaluation
            judge_evaluation_batch.append({
                'sample_idx': sample_idx,
                'question_idx': question_idx,
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'conv_id': batch_data[sample_idx]['conv_id'],
                'chunk_id': batch_data[sample_idx]['chunk_id']
            })
        
        # Step 5: Evaluate ALL answers at once using batched judge LLM call
        print(f"Evaluating {len(judge_evaluation_batch)} answers with {'LLM judge' if self.config.use_llm_judge else 'string matching'}")
        all_judge_scores = self._evaluate_answers_with_judge_batched(
            judge_evaluation_batch, 
            use_llm_judge=self.config.use_llm_judge
        )
        
        # Step 6: Organize rewards and print grouped by (conv_id, question_idx)
        # Collect all results with scores
        results_with_scores = []
        for eval_item, judge_score in zip(judge_evaluation_batch, all_judge_scores):
            sample_idx = eval_item['sample_idx']
            full_answer_rewards[sample_idx].append(judge_score)
            results_with_scores.append((eval_item, judge_score))
        
        # Sort by (conv_id, question_idx, sample_idx) to group all rollouts of same question together
        # Just for clearer printing, to have all rollouts of conv-48 Q0, then conv-48 Q1, etc.
        results_with_scores.sort(key=lambda x: (x[0]['conv_id'], x[0]['question_idx'], x[0]['sample_idx']))
        
        # Print in sorted order (all rollouts of conv-48 Q0, then conv-48 Q1, etc.)
        for eval_item, judge_score in results_with_scores:
            print(f"Conv {eval_item['conv_id']}, chunk {eval_item['chunk_id']}, Q{eval_item['question_idx']}: "
                  f"Question: '{eval_item['question'][:50]}', "
                  f"Predicted: '{eval_item['predicted_answer'][:50]}', "
                  f"Gold: '{eval_item['gold_answer'][:50]}', "
                  f"{'Judge' if self.config.use_llm_judge else 'Match'} Score: {judge_score:.3f}")
        
        # Step 7: Add to all_inputs for later concatenation
        # With variable QA counts, we organize by sample instead of by question
        # The combine function will handle per-sample concatenation and then batch padding
        
        # Group QA results back by sample
        sample_qa_prompts = [[] for _ in range(len(batch_data))]
        sample_qa_prompt_masks = [[] for _ in range(len(batch_data))]
        sample_qa_responses = [[] for _ in range(len(batch_data))]
        sample_qa_response_masks = [[] for _ in range(len(batch_data))]
        
        for flat_idx, (sample_idx, question_idx) in enumerate(qa_metadata):
            sample_qa_prompts[sample_idx].append(qa_input_ids[flat_idx])
            sample_qa_prompt_masks[sample_idx].append(qa_attention_masks[flat_idx])
            sample_qa_responses[sample_idx].append(qa_response_batch.batch['input_ids'][flat_idx, total_prompt_length_of_qa:])
            sample_qa_response_masks[sample_idx].append(qa_response_batch.batch['attention_mask'][flat_idx, total_prompt_length_of_qa:])
        
        # Now add QAs in interleaved format for each sample
        # We need to maintain the structure: [memory_prompt, memory_response, qa_prompt_1, qa_response_1, ...]
        # Currently all_inputs_ids has [memory_prompt, memory_response] already added
        # We need to add the QA pairs in an interleaved manner
        
        # Find max number of QAs across batch
        max_qa_count = max(sampled_qa_counts)
        
        # Find a sample with at least one QA to use as template for empty tensors
        template_sample_idx = None
        for idx, count in enumerate(sampled_qa_counts):
            if count > 0:
                template_sample_idx = idx
                break
        
        if template_sample_idx is None:
            raise ValueError("All samples have 0 QAs - cannot create batch structure")
        
        # Add QA pairs question by question (to maintain interleaved structure)
        for qa_idx in range(max_qa_count):
            qa_prompts_batch = []
            qa_prompt_masks_batch = []
            qa_responses_batch = []
            qa_response_masks_batch = []
            
            for sample_idx in range(len(batch_data)):
                if qa_idx < sampled_qa_counts[sample_idx]:
                    # Sample has this QA
                    qa_prompts_batch.append(sample_qa_prompts[sample_idx][qa_idx])
                    qa_prompt_masks_batch.append(sample_qa_prompt_masks[sample_idx][qa_idx])
                    qa_responses_batch.append(sample_qa_responses[sample_idx][qa_idx])
                    qa_response_masks_batch.append(sample_qa_response_masks[sample_idx][qa_idx])
                else:
                    # Sample doesn't have this QA - create empty tensors with same shape as template
                    # Use template_sample_idx (guaranteed to have ≥1 QA) for shape reference
                    # These have attention_mask=0, so combine function will skip them
                    empty_prompt = torch.full_like(sample_qa_prompts[template_sample_idx][0], self.tokenizer.pad_token_id)
                    empty_mask = torch.zeros_like(sample_qa_prompt_masks[template_sample_idx][0])
                    qa_prompts_batch.append(empty_prompt)
                    qa_prompt_masks_batch.append(empty_mask)
                    
                    empty_response = torch.full_like(sample_qa_responses[template_sample_idx][0], self.tokenizer.pad_token_id)
                    empty_response_mask = torch.zeros_like(sample_qa_response_masks[template_sample_idx][0])
                    qa_responses_batch.append(empty_response)
                    qa_response_masks_batch.append(empty_response_mask)
            
            all_inputs_ids.append(torch.stack(qa_prompts_batch))
            all_attention_masks.append(torch.stack(qa_prompt_masks_batch))
            all_inputs_ids.append(torch.stack(qa_responses_batch))
            all_attention_masks.append(torch.stack(qa_response_masks_batch))

        # Verify structure
        assert len(full_answer_rewards) == len(batch_data), "Full answer rewards length must match batch size"
        for i in range(len(batch_data)):
            assert len(full_answer_rewards[i]) == sampled_qa_counts[i], \
                f"Sample {i}: rewards count {len(full_answer_rewards[i])} != sampled QA count {sampled_qa_counts[i]}"

        assert len(all_inputs_ids) == len(all_attention_masks), "Inputs and attention masks count must match"
        expected_total_inputs = 2 + 2 * max_qa_count  # memory (2) + QA pairs (2 each, padded to max)
        assert len(all_inputs_ids) == expected_total_inputs, \
            f"Total inputs {len(all_inputs_ids)} != expected {expected_total_inputs} (2 memory + 2*{max_qa_count} QAs)"

        # ========================================
        # GRPO-AWARE CACHING: Select Best Rollout
        # ========================================
        # With GRPO, we have multiple rollouts of the same (conv_id, chunk_id) pair.
        # We need to cache only ONE memory state per (conv_id, chunk_id) for the next chunk to load.
        # Strategy: Cache the rollout with the highest total reward (format + QA average)
        
        # Compute total reward for each sample: format_reward + average(QA_rewards)
        total_rewards = []
        qa_rewards_averaged = []  # Store for passing to reward manager
        for i in range(len(batch_data)):
            # if no QA rewards (possible now), average is 0, we only rely on format reward in this case
            qa_reward_avg = np.mean(full_answer_rewards[i]) if full_answer_rewards[i] else 0.0
            qa_rewards_averaged.append(qa_reward_avg)
            total_reward = self.config.format_reward_weight * format_rewards[i] + self.config.answer_reward_weight * qa_reward_avg \
                            + self.config.memory_reward_weight * memory_rewards[i]
            total_rewards.append(total_reward)
        
        # Group rollouts by (conv_id, chunk_id) to find best rollout per conversation
        conv_chunk_to_indices = {}
        for idx, data in enumerate(batch_data):
            key = (data['conv_id'], data['chunk_id'])
            if key not in conv_chunk_to_indices:
                conv_chunk_to_indices[key] = []
            conv_chunk_to_indices[key].append(idx)
        
        # For each unique (conv_id, chunk_id), cache only the best rollout's memory state
        for (conv_id, chunk_id), indices in conv_chunk_to_indices.items():
            if len(indices) == 1:
                # Single rollout - just cache it
                idx = indices[0]
                data = batch_data[idx]
                # data['manager'].cache_snapshot(data['memory'], conv_id, chunk_id, epoch, self.split)
                print(f"[ON-POLICY] Cached memory state for conv {conv_id}, chunk {chunk_id}, epoch {epoch} "
                      f"(single rollout, reward={total_rewards[idx]:.3f})")
            else:
                # Multiple rollouts (GRPO) - cache only the best one
                rollout_rewards = [total_rewards[idx] for idx in indices]
                best_rollout_idx = indices[np.argmax(rollout_rewards)]
                best_data = batch_data[best_rollout_idx]
                # best_data['manager'].cache_snapshot(best_data['memory'], conv_id, chunk_id, epoch, self.split)
                print(f"[ON-POLICY + GRPO] Cached BEST memory state for conv {conv_id}, chunk {chunk_id}, epoch {epoch} "
                      f"(selected rollout {indices.index(best_rollout_idx)+1}/{len(indices)} with reward={total_rewards[best_rollout_idx]:.3f}, "
                      f"all rewards={[f'{r:.3f}' for r in rollout_rewards]})")


        full_input_ids, full_attention_mask, full_position_ids, full_response_mask, isolated_response_ids, isolated_response_attention_mask, isolated_prompt_ids = self.combine_prompts_and_responses_for_rl_training(all_inputs_ids, all_attention_masks)
        assert full_input_ids.shape == full_attention_mask.shape == full_position_ids.shape == full_response_mask.shape, "Full sequence tensors must have the same shape"
        assert isolated_response_ids.shape == isolated_response_attention_mask.shape, "Isolated response tensors must have the same shape"
        
        # print(f"[run_chunks] Final processed shapes:")
        # print(f"  - full_input_ids: {full_input_ids.shape}")
        # print(f"  - full_attention_mask: {full_attention_mask.shape}")
        # print(f"  - full_position_ids: {full_position_ids.shape}")
        # print(f"  - full_response_mask: {full_response_mask.shape}")
        # print(f"  - isolated_response_ids: {isolated_response_ids.shape}")
        # print(f"  - isolated_response_attention_mask: {isolated_response_attention_mask.shape}")
        # print(f"  - isolated_prompt_ids: {isolated_prompt_ids.shape}")
        # print(f"  - format_rewards length: {len(format_rewards)}")
        # print(f"  - full_answer_rewards length: {len(full_answer_rewards)}")

        # Normalize rewards if needed across the batch
        epsilon = 1e-8
        total_rewards = np.array(total_rewards)
        mu = np.mean(total_rewards)
        sigma = np.std(total_rewards) + epsilon
        total_rewards_normalized = (total_rewards - mu) / sigma
        
        # make a data proto to return 
        # Store rewards in non_tensor_batch (NOT meta_info) to ensure proper padding/unpadding
        # meta_info should only contain global scalars, not per-sample arrays
        result = DataProto.from_dict({
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask, # attention on full sequence
            'position_ids': full_position_ids, 
            'responses': isolated_response_ids, # Only the responses ids
            'response_mask': isolated_response_attention_mask, # Mask for the responses ids only as they have right padding
            'prompts': isolated_prompt_ids, # Only the prompts ids (left-padded)
        }, meta_info=gen_batch.meta_info)
        
        # Add rewards to non_tensor_batch for proper handling during padding/unpadding
        # These will be automatically handled by unpad_dataproto() and can be chunked safely
        result.non_tensor_batch = {
            'format_rewards': np.array(format_rewards, dtype=object),  # Array of floats
            'full_answer_rewards': np.array(full_answer_rewards, dtype=object),  # Array of lists
            'qa_rewards_averaged': np.array(qa_rewards_averaged, dtype=object),  # Array of floats
            'memory_rewards': np.array(memory_rewards, dtype=object),  # Array of floats
            'total_rewards_unnormalized': np.array(total_rewards, dtype=object),  # Array of floats (weighted total)
            'total_rewards': np.array(total_rewards_normalized, dtype=object),  # Array of floats (weighted total)
        }
        
        return result
        