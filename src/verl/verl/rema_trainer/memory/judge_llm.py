# judge_llm.py
import os
import json
import hashlib
import threading
# from openai import OpenAI
import google.generativeai as genai
from filelock import FileLock
from typing import List

# --------------------------
# Configuration
# --------------------------
CACHE_DIR = os.getenv("OPENAI_CACHE_DIR", ".")  # Keep same cache dir for compatibility
MAIN_CACHE_FILE = os.path.join(CACHE_DIR, "judge_cache.json")
PROCESS_CACHE_FILE = os.path.join(CACHE_DIR, f"judge_cache_{os.getpid()}.json")
LOCK_FILE = MAIN_CACHE_FILE + ".lock"

# Thread-safety lock for JUDGE_CACHE (protects against concurrent thread access within same process)
_CACHE_LOCK = threading.Lock()

# Load per-process cache
if os.path.exists(PROCESS_CACHE_FILE):
    with open(PROCESS_CACHE_FILE, "r") as f:
        JUDGE_CACHE = json.load(f)
else:
    JUDGE_CACHE = {}

# OpenAI client (commented out)
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Gemini client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Small, fast model

# --------------------------
# Utility functions
# --------------------------
def _hash_prompt(prompt: str) -> str:
    """Stable hash so slight whitespace differences produce different keys."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def _save_process_cache():
    """Save per-process cache to disk (non-blocking)."""
    with open(PROCESS_CACHE_FILE, "w") as f:
        json.dump(JUDGE_CACHE, f, indent=2)

def merge_to_main_cache():
    """Merge per-process cache into the main cache safely and sync back."""
    global JUDGE_CACHE
    
    # Thread-safe: acquire lock before reading JUDGE_CACHE
    with _CACHE_LOCK:
        # Don't merge if we have no new entries
        if not JUDGE_CACHE:
            return
        
        # Make a snapshot to avoid holding lock during file I/O
        cache_snapshot = dict(JUDGE_CACHE)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with FileLock(LOCK_FILE, timeout=60):
                # Read current main cache
                if os.path.exists(MAIN_CACHE_FILE):
                    try:
                        with open(MAIN_CACHE_FILE, "r") as f:
                            main_cache = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"[merge error] Corrupted main cache on attempt {attempt+1}, creating backup: {e}")
                        # Backup corrupted file
                        backup_file = MAIN_CACHE_FILE + ".corrupted." + str(os.getpid())
                        if os.path.exists(MAIN_CACHE_FILE):
                            try:
                                os.rename(MAIN_CACHE_FILE, backup_file)
                            except:
                                pass
                        main_cache = {}
                else:
                    main_cache = {}
                
                # Merge new entries from this process (using snapshot, not live dict)
                original_size = len(main_cache)
                main_cache.update(cache_snapshot)
                new_size = len(main_cache)
                
                # Atomic write with validation
                tmp_file = MAIN_CACHE_FILE + ".tmp." + str(os.getpid())
                try:
                    with open(tmp_file, "w") as f:
                        json.dump(main_cache, f, indent=2, ensure_ascii=False)
                    
                    # Validate the temp file before replacing
                    with open(tmp_file, "r") as f:
                        validated = json.load(f)
                        if len(validated) != new_size:
                            raise ValueError(f"Size mismatch: expected {new_size}, got {len(validated)}")
                    
                    # Atomic replace
                    os.replace(tmp_file, MAIN_CACHE_FILE)
                    
                    # Verify the write was successful
                    with open(MAIN_CACHE_FILE, "r") as f:
                        final_check = json.load(f)
                        if len(final_check) != new_size:
                            raise ValueError(f"Write verification failed: expected {new_size}, got {len(final_check)}")
                    
                    print(f"[merge success] Merged {new_size - original_size} new entries (total: {new_size})")
                    
                    # Update our local cache with merged result (thread-safe)
                    with _CACHE_LOCK:
                        JUDGE_CACHE.update(main_cache)
                        _save_process_cache()
                    return  # Success!
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[merge error] Validation failed on attempt {attempt+1}: {e}")
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except:
                            pass
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        print(f"[merge error] Failed after {max_retries} attempts, keeping process cache only")
                        return
                        
        except Exception as e:
            print(f"[merge error] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5 * (attempt + 1))
                continue
            else:
                print(f"[merge error] All {max_retries} attempts failed")
                return

# Counter for new entries
_NEW_ENTRIES_COUNT = 0
MERGE_EVERY_N = 20  # merge after every 20 new entries (reduce lock contention)


def judge_with_llm(prompt: str) -> str:
    """
    Call Gemini Flash model deterministically and cache the output.
    Merges per-process cache to main cache every N new entries.
    
    Note: Name kept as 'judge_with_llm' for backward compatibility,
    but actually uses Gemini API now.
    """
    global _NEW_ENTRIES_COUNT

    key = _hash_prompt(prompt)

    # Thread-safe: check cache with lock
    with _CACHE_LOCK:
        if key in JUDGE_CACHE:
            return JUDGE_CACHE[key]

    # Call Gemini API (outside lock to avoid blocking other threads)
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=1.0,
        )
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        result = response.text.strip()
    except Exception as e:
        print(f"[judge error] {e}")
        result = ""

    # Thread-safe: save to cache with lock
    with _CACHE_LOCK:
        JUDGE_CACHE[key] = result
        _save_process_cache()  # non-blocking save

        # Increment counter and merge if threshold reached
        _NEW_ENTRIES_COUNT += 1
        # if _NEW_ENTRIES_COUNT >= MERGE_EVERY_N:
        #     merge_to_main_cache()
        #     _NEW_ENTRIES_COUNT = 0

    return result


def judge_with_llm_batch(prompts: List[str]) -> List[str]:
    """
    Call Gemini Flash model for multiple prompts with concurrent processing.
    Uses caching and concurrent API calls for efficiency.
    
    Args:
        prompts: List of prompt strings to evaluate
        
    Returns:
        List of response strings (same order as input prompts)
    """
    global _NEW_ENTRIES_COUNT
    
    if not prompts:
        return []
    
    # Check cache first
    results = [None] * len(prompts)
    uncached_indices = []
    uncached_prompts = []
    
    for idx, prompt in enumerate(prompts):
        key = _hash_prompt(prompt)
        if key in JUDGE_CACHE:
            results[idx] = JUDGE_CACHE[key]
        else:
            uncached_indices.append(idx)
            uncached_prompts.append(prompt)
    
    # Process uncached prompts concurrently
    if uncached_prompts:
        print(f"[judge_with_llm_batch] Processing {len(uncached_prompts)} uncached prompts concurrently...")
        
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=1.0,
        )
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def call_api(idx_prompt_pair):
            """Helper function for concurrent API calls"""
            orig_idx, prompt = idx_prompt_pair
            try:
                response = gemini_model.generate_content(prompt, generation_config=generation_config)
                response_text = response.text.strip()
                return orig_idx, prompt, response_text, None
            except Exception as e:
                return orig_idx, prompt, "", str(e)
        
        # Use ThreadPoolExecutor for concurrent API calls
        new_entries = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(call_api, (orig_idx, prompt)): orig_idx 
                      for orig_idx, prompt in zip(uncached_indices, uncached_prompts)}
            
            for future in as_completed(futures):
                orig_idx, prompt, response_text, error = future.result()
                
                if error:
                    print(f"[judge_with_llm_batch] API error for idx {orig_idx}: {error}")
                    results[orig_idx] = ""
                else:
                    # Cache the response
                    key = _hash_prompt(prompt)
                    JUDGE_CACHE[key] = response_text
                    results[orig_idx] = response_text
                    new_entries += 1
        
        # Update global counter after all processing
        _NEW_ENTRIES_COUNT += new_entries
        
        # Save process cache once after all responses
        _save_process_cache()
        
        # Merge to main cache if threshold reached
        # if _NEW_ENTRIES_COUNT >= MERGE_EVERY_N:
        #     merge_to_main_cache()
        #     _NEW_ENTRIES_COUNT = 0
        
        print(f"[judge_with_llm_batch] Completed: {len(prompts) - len(uncached_prompts)} cache hits, {len(uncached_prompts)} API calls")
    
    return results


# OpenAI implementation (commented out for reference)
# def judge_with_llm_openai(prompt: str) -> str:
#     """
#     Call GPT-4.1-mini deterministically and cache the output.
#     """
#     global _NEW_ENTRIES_COUNT
#     key = _hash_prompt(prompt)
#     if key in JUDGE_CACHE:
#         return JUDGE_CACHE[key]
#     
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             top_p=1.0,
#         )
#         result = response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"[judge error] {e}")
#         result = ""
#     
#     JUDGE_CACHE[key] = result
#     _save_process_cache()
#     _NEW_ENTRIES_COUNT += 1
#     if _NEW_ENTRIES_COUNT >= MERGE_EVERY_N:
#         merge_to_main_cache()
#         _NEW_ENTRIES_COUNT = 0
#     return result