"""
Teacher Model for Memory Snapshot Generation

This module provides a strong, frozen teacher model (Gemini) to generate
high-quality memory snapshots at epoch start. These snapshots serve as
expert demonstrations for the RL training policy to learn from.

Key Features:
- Uses Gemini API (strong model) instead of training policy
- Batched generation for efficiency
- Automatic caching to avoid redundant API calls
- Independent from training policy (no on-policy leakage)

Usage:
    teacher = TeacherModel()
    teacher.generate_memory_snapshots_batch(dataset, epoch)
"""

import os
import json
import hashlib
import time
from google import genai
from google.genai import types
from typing import List, Dict, Any
from pathlib import Path
from filelock import FileLock
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.memory_core.prompt_generator import generate_memory_prompt
from verl.rema_trainer.memory.utils.parse_response import extract_llm_json_from_response


class TeacherModel:
    """
    Strong teacher model (Gemini) for generating expert memory snapshots.
    
    This model generates high-quality memory operations that serve as
    demonstrations for the RL policy to learn from. It operates independently
    from the training policy to avoid on-policy leakage.
    """
    
    def __init__(
        self, 
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = None,
        top_k_memories: int = 20,
        similarity_threshold: float = 0.1,
        temperature: float = 0.0,
    ):
        """
        Initialize teacher model.
        
        Args:
            model_name: Gemini model to use (default: gemini-1.5-flash, fast and good)
            cache_dir: Directory for caching API responses
            top_k_memories: Number of memories to show in prompts
            similarity_threshold: Similarity threshold for memory retrieval
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model_name = model_name
        self.top_k_memories = top_k_memories
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        
        # Setup cache
        self.cache_dir = cache_dir or os.getenv("TEACHER_CACHE_DIR", "./teacher_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "teacher_responses.json")
        self.lock_file = self.cache_file + ".lock"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Configure Gemini with new SDK
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name_full = f"models/{model_name}" if not model_name.startswith("models/") else model_name
        
        print(f"[TeacherModel] Initialized with {model_name}")
        print(f"[TeacherModel] Using new Gemini batch API")
        print(f"[TeacherModel] Cache: {self.cache_file} ({len(self.cache)} entries)")
    
    def _load_cache(self) -> Dict[str, str]:
        """Load cache from disk with file locking."""
        try:
            with FileLock(self.lock_file, timeout=10):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, 'r') as f:
                        return json.load(f)
        except Exception as e:
            print(f"[TeacherModel] Cache load error: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk with file locking."""
        try:
            with FileLock(self.lock_file, timeout=10):
                tmp_file = self.cache_file + ".tmp"
                with open(tmp_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                os.replace(tmp_file, self.cache_file)
        except Exception as e:
            print(f"[TeacherModel] Cache save error: {e}")
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create stable hash for prompt caching."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    
    def generate_memory_operations(
        self, 
        memory: Memory, 
        turns: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate memory operations for given turns using teacher model.
        
        Args:
            memory: Current memory state
            turns: Conversation turns to process
            use_cache: Whether to use cached responses
            
        Returns:
            Dict with 'operations' list and '_parse_success' flag
        """
        # Generate prompt
        prompt, _, _ = generate_memory_prompt(
            memory,
            turns,
            top_k_memories=self.top_k_memories,
            similarity_threshold=self.similarity_threshold,
            use_similarity=True
        )
        
        # Check cache
        prompt_hash = self._hash_prompt(prompt)
        if use_cache and prompt_hash in self.cache:
            response_text = self.cache[prompt_hash]
            # print(f"[TeacherModel] Cache hit for prompt hash {prompt_hash[:8]}")
        else:
            # Call Gemini API (single request)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name_full,
                    contents=[{
                        'parts': [{'text': prompt}],
                        'role': 'user'
                    }],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=1.0,
                    )
                )
                response_text = response.text.strip()
                
                # Cache the response
                self.cache[prompt_hash] = response_text
                self._save_cache()
                # print(f"[TeacherModel] API call + cache save for prompt hash {prompt_hash[:8]}")
            except Exception as e:
                print(f"[TeacherModel] API error: {e}")
                response_text = '{"operations": []}'
        
        # Parse response
        return extract_llm_json_from_response(response_text)
    
    def generate_memory_operations_batch(
        self,
        batch_data: List[Dict[str, Any]],
        use_cache: bool = True,
        poll_interval: int = 5,
        max_wait: int = 600
    ) -> List[Dict[str, Any]]:
        """
        Generate memory operations for a batch of conversations using native Gemini batch API.
        
        This uses Google's native batch processing:
        - Cached prompts are returned immediately
        - Uncached prompts are sent in ONE batched API call to Gemini
        - Polls for completion and retrieves results
        
        Args:
            batch_data: List of dicts with 'memory', 'turns', 'conv_id', 'chunk_id'
            use_cache: Whether to use cached responses
            poll_interval: Seconds between polling batch job status
            max_wait: Maximum seconds to wait for batch completion
            
        Returns:
            List of response dicts (same order as input)
        """
        # Step 1: Generate all prompts and check cache
        prompts_and_hashes = []
        results = [None] * len(batch_data)  # Pre-allocate results list
        uncached_indices = []
        batch_requests = []
        cache_hits = 0
        
        for idx, data in enumerate(batch_data):
            # Generate prompt
            prompt, _, _ = generate_memory_prompt(
                data['memory'],
                data['turns'],
                top_k_memories=self.top_k_memories,
                similarity_threshold=self.similarity_threshold,
                use_similarity=True
            )
            prompt_hash = self._hash_prompt(prompt)
            prompts_and_hashes.append((prompt, prompt_hash))
            
            # Check cache
            if use_cache and prompt_hash in self.cache:
                response_text = self.cache[prompt_hash]
                results[idx] = extract_llm_json_from_response(response_text)
                cache_hits += 1
            else:
                uncached_indices.append(idx)
                batch_requests.append({
                    'contents': [{
                        'parts': [{'text': prompt}],
                        'role': 'user'
                    }]
                })
        
        # Step 2: Process uncached prompts with native batch API
        if batch_requests:
            print(f"[TeacherModel] Submitting {len(batch_requests)} requests as native Gemini batch...")
            
            try:
                # Submit batch job
                batch_job = self.client.batches.create(
                    model=self.model_name_full,
                    src=batch_requests,
                    config={
                        'display_name': f'teacher-batch-{int(time.time())}',
                    }
                )
                
                print(f"[TeacherModel] Batch job created: {batch_job.name}")
                print(f"[TeacherModel] Waiting for completion (polling every {poll_interval}s, max {max_wait}s)...")
                
                # Poll for completion
                elapsed = 0
                while elapsed < max_wait:
                    job_status = self.client.batches.get(name=batch_job.name)
                    
                    if job_status.state == 'JOB_STATE_SUCCEEDED':
                        print(f"[TeacherModel] ✓ Batch completed successfully after {elapsed}s")
                        
                        # Retrieve results
                        for req_idx, orig_idx in enumerate(uncached_indices):
                            try:
                                # Get the response for this request
                                response = job_status.responses[req_idx]
                                if response and hasattr(response, 'candidates') and response.candidates:
                                    response_text = response.candidates[0].content.parts[0].text.strip()
                                else:
                                    response_text = '{"operations": []}'
                                
                                # Cache the response
                                _, prompt_hash = prompts_and_hashes[orig_idx]
                                self.cache[prompt_hash] = response_text
                                
                                # Parse and store result
                                results[orig_idx] = extract_llm_json_from_response(response_text)
                            except Exception as e:
                                print(f"[TeacherModel] Error parsing result {req_idx}: {e}")
                                results[orig_idx] = {"operations": [], "_parse_success": False}
                        
                        # Save cache once after all responses
                        self._save_cache()
                        print(f"[TeacherModel] Saved {len(batch_requests)} new responses to cache")
                        break
                    
                    elif job_status.state in ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
                        print(f"[TeacherModel] ✗ Batch job {job_status.state}")
                        # Fill with empty operations
                        for orig_idx in uncached_indices:
                            results[orig_idx] = {"operations": [], "_parse_success": False}
                        break
                    
                    else:
                        # Still processing
                        time.sleep(poll_interval)
                        elapsed += poll_interval
                
                if elapsed >= max_wait:
                    print(f"[TeacherModel] ✗ Batch timeout after {max_wait}s")
                    # Fill with empty operations
                    for orig_idx in uncached_indices:
                        results[orig_idx] = {"operations": [], "_parse_success": False}
                        
            except Exception as e:
                print(f"[TeacherModel] Batch API error: {e}")
                # Fallback: Fill with empty operations
                for orig_idx in uncached_indices:
                    results[orig_idx] = {"operations": [], "_parse_success": False}
        
        # Safety check: Replace any remaining None results with empty operations
        for i in range(len(results)):
            if results[i] is None:
                print(f"[TeacherModel] WARNING: Result {i} is None, using empty operations")
                results[i] = {"operations": [], "_parse_success": False}
        
        print(f"[TeacherModel] Batch complete: {cache_hits} cache hits, {len(batch_requests)} batch API calls")
        return results
    
    def generate_memory_snapshots_for_dataset(
        self,
        dataset,
        epoch: int,
        split: str = "train",
    ):
        """
        Generate memory snapshots for entire dataset using teacher model.
        
        This is the main entry point for pre-generating expert memory snapshots
        at epoch start. It processes conversations sequentially (chunk by chunk)
        and caches the resulting memory states.
        
        Args:
            dataset: Dataset containing conversation chunks
            epoch: Current epoch number
            split: Dataset split ('train' or 'validation')
        """
        print(f"\n{'='*60}")
        print(f"[TeacherModel] Generating expert memory snapshots")
        print(f"  Model: {self.model_name}")
        print(f"  Epoch: {epoch}")
        print(f"  Split: {split}")
        print(f"{'='*60}\n")
        
        # Group dataset by conversation
        conv_dict = {}
        for item in dataset:
            conv_id = item.get("sample_id", None)
            if conv_id is None:
                raise ValueError("Each dataset item must have a 'sample_id'")
            
            if conv_id not in conv_dict:
                conv_dict[conv_id] = []
            conv_dict[conv_id].append(item)
        
        # Sort by chunk_id
        for conv_id in conv_dict:
            conv_dict[conv_id] = sorted(conv_dict[conv_id], key=lambda x: x.get("chunk_id", -1))
        
        # Get cache directory
        if split == "train":
            cache_dir = os.getenv("MEMORY_CACHE_DIR", "./memory_cache")
        else:
            cache_dir = os.getenv("MEMORY_CACHE_DIR_VAL", "./memory_cache_val")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check if all snapshots are already cached
        all_cached = True
        for conv_id, items in conv_dict.items():
            for item in items:
                chunk_id = item.get("chunk_id")
                cache_file_pkl = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.pkl")
                cache_file_json = os.path.join(cache_dir, f"epoch_{epoch}", conv_id, f"chunk_{chunk_id}.json")
                if not (os.path.exists(cache_file_pkl) or os.path.exists(cache_file_json)):
                    all_cached = False
                    break
            if not all_cached:
                break
        
        if all_cached:
            print(f"✓ All teacher memory snapshots already cached for epoch {epoch} ({split}). Skipping generation.")
            return
        
        # Initialize managers and memories
        conv_managers = {conv_id: MemoryManager() for conv_id in conv_dict}
        conv_memories = {conv_id: Memory() for conv_id in conv_dict}
        
        # Process chunk by chunk
        max_chunks = max(len(items) for items in conv_dict.values())
        
        for chunk_idx in range(max_chunks):
            print(f"\n--- Processing chunk index {chunk_idx}/{max_chunks-1} ---")
            
            batch_data = []
            for conv_id, items in conv_dict.items():
                if chunk_idx < len(items):
                    item = items[chunk_idx]
                    chunk_id = item.get("chunk_id")
                    turns = item.get("turns_json")
                    
                    if isinstance(turns, str):
                        turns = json.loads(turns)
                    
                    batch_data.append({
                        'conv_id': conv_id,
                        'chunk_id': chunk_id,
                        'turns': turns,
                        'memory': conv_memories[conv_id],
                        'manager': conv_managers[conv_id],
                    })
            
            if not batch_data:
                continue
            
            print(f"Processing {len(batch_data)} conversations...")
            
            # Generate operations using teacher model
            results = self.generate_memory_operations_batch(batch_data)
            
            # Execute operations and cache snapshots
            for data, result in zip(batch_data, results):
                # Handle None results (API failures)
                if result is None:
                    print(f"[TeacherModel] ERROR: No result for conv {data['conv_id']}, chunk {data['chunk_id']} - using empty operations")
                    operations = []
                    json_success = False
                else:
                    operations = result.get("operations", [])
                    json_success = result.get("_parse_success", False)
                
                if not json_success and result is not None:
                    print(f"[TeacherModel] WARNING: JSON parse failed for conv {data['conv_id']}, chunk {data['chunk_id']}")
                    # Continue with empty operations
                    operations = []
                
                # Attach metadata and execute
                operations = data['manager'].attach_turn_metadata_to_operations(
                    operations, data['turns'], data['conv_id']
                )
                
                exec_result = data['manager'].execute_batch(data['memory'], operations)
                
                # Cache the snapshot
                data['manager'].cache_snapshot(
                    data['memory'], 
                    data['conv_id'], 
                    data['chunk_id'], 
                    epoch, 
                    split
                )
                
                print(f"  ✓ Conv {data['conv_id']}, chunk {data['chunk_id']}: "
                      f"{exec_result['successful']}/{exec_result['total_commands']} ops, "
                      f"{len(data['memory'].memories)} total memories")
        
        print(f"\n{'='*60}")
        print(f"[TeacherModel] ✓ Completed snapshot generation for epoch {epoch}")
        print(f"  Cache: {cache_dir}/epoch_{epoch}/")
        print(f"{'='*60}\n")


# Convenience function for backward compatibility
def create_teacher_model(**kwargs) -> TeacherModel:
    """
    Create a teacher model instance.
    
    Args:
        **kwargs: Additional arguments passed to TeacherModel
        
    Returns:
        TeacherModel instance
    """
    return TeacherModel(**kwargs)
