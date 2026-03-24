# judge_llm.py
import os
import json
import hashlib
import threading
import google.generativeai as genai
from together import Together
from filelock import FileLock
from typing import List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --------------------------
# Configuration
# --------------------------
# API selection: set JUDGE_PROVIDER to one of {gemini, together, openai}
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "").strip().lower()
if JUDGE_PROVIDER not in {"gemini", "together", "openai"}:
    raise ValueError(
        "JUDGE_PROVIDER must be set to one of: gemini, together, openai"
    )

GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash-lite")
TOGETHER_JUDGE_MODEL = os.getenv("TOGETHER_JUDGE_MODEL", "openai/gpt-oss-120b")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

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

# Initialize clients based on selected provider
if JUDGE_PROVIDER == "gemini":
    print("[judge_llm] Using Gemini API")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(GEMINI_JUDGE_MODEL)
    together_clients = None
    openai_client = None
elif JUDGE_PROVIDER == "together":
    print("[judge_llm] Using Together AI API")
    _together_keys = [k.strip() for k in os.environ.get("TOGETHER_API_KEY", "").split(",") if k.strip()]
    if not _together_keys:
        together_clients = [Together()]
    else:
        together_clients = [Together(api_key=k) for k in _together_keys]
    gemini_model = None
    openai_client = None
elif JUDGE_PROVIDER == "openai":
    if OpenAI is None:
        raise ImportError("OpenAI provider selected but openai package is not installed. Install with: pip install openai>=1.5.0")
    # JUDGE_BASE_URLS: comma-separated list of base URLs for round-robin across multiple servers
    _judge_api_key = os.environ.get("JUDGE_API_KEY", "EMPTY")
    _base_urls = [u.strip() for u in os.environ.get("JUDGE_BASE_URLS", "").split(",") if u.strip()]
    if not _base_urls:
        raise ValueError("JUDGE_PROVIDER='openai' requires JUDGE_BASE_URLS to be set to a non-empty comma-separated list of server URLs.")
    openai_clients = [OpenAI(api_key=_judge_api_key, base_url=url) for url in _base_urls]
    print(f"[judge_llm] Using OpenAI provider with {len(openai_clients)} server(s)")
    gemini_model = None
    together_clients = None
else:
    raise ValueError(f"Unsupported JUDGE_PROVIDER='{JUDGE_PROVIDER}'. Expected one of: gemini, together, openai")


def _call_judge_api(prompt: str, attempt: int = 0) -> str:
    """Call selected provider with deterministic generation settings."""
    if JUDGE_PROVIDER == "gemini":
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=1.0,
        )
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return (response.text or "").strip()

    if JUDGE_PROVIDER == "together":
        client_idx = attempt % len(together_clients)
        response = together_clients[client_idx].chat.completions.create(
            model=TOGETHER_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
        )
        return (response.choices[0].message.content or "").strip()

    if JUDGE_PROVIDER == "openai":
        client_idx = attempt % len(openai_clients)
        response = openai_clients[client_idx].chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1.0,
        )
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning", None) or ""
        return content.strip()

    raise ValueError(f"Unsupported JUDGE_PROVIDER='{JUDGE_PROVIDER}'")


def _get_max_workers() -> int:
    """Provider-aware default concurrency with env override."""
    env_workers = os.getenv("JUDGE_LLM_MAX_WORKERS")
    if env_workers:
        try:
            return max(1, int(env_workers))
        except ValueError:
            pass

    if JUDGE_PROVIDER == "gemini":
        return 10
    if JUDGE_PROVIDER == "together":
        return max(1, len(together_clients))
    if JUDGE_PROVIDER == "openai":
        return max(1, len(openai_clients))
    return 8

# --------------------------
# Utility functions
# --------------------------
def _hash_prompt(prompt: str) -> str:
    """Stable hash so slight whitespace differences produce different keys."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def _save_process_cache():
    """Save per-process cache to disk (non-blocking)."""
    with open(PROCESS_CACHE_FILE, "w") as f:
        json.dump(JUDGE_CACHE, f, separators=(',', ':'))

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
MERGE_EVERY_N = 100  # merge after every 20 new entries (reduce lock contention)


def judge_with_llm(prompt: str) -> str:
    """
    Call selected LLM provider deterministically and cache the output.
    Merges per-process cache to main cache every N new entries.
    """
    global _NEW_ENTRIES_COUNT

    key = _hash_prompt(prompt)

    # Thread-safe: check cache with lock
    with _CACHE_LOCK:
        if key in JUDGE_CACHE:
            return JUDGE_CACHE[key]

    # Call appropriate API (outside lock to avoid blocking other threads)
    import time
    max_retries = 10
    result = ""
    for attempt in range(max_retries):
        try:
            result = _call_judge_api(prompt, attempt=attempt)
            break
        except Exception as e:
            print(f"[judge error] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1)  # tiny sleep to quickly try next key
            else:
                result = ""

    # Thread-safe: save to cache with lock
    with _CACHE_LOCK:
        JUDGE_CACHE[key] = result

        # Increment counter and save periodically (not on every entry)
        _NEW_ENTRIES_COUNT += 1
        if _NEW_ENTRIES_COUNT >= MERGE_EVERY_N:
            _save_process_cache()
            _NEW_ENTRIES_COUNT = 0

    return result


def judge_with_llm_batch(prompts: List[str]) -> List[str]:
    """
    Call the selected LLM provider for multiple prompts with concurrent processing.
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

    with _CACHE_LOCK:
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
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def call_api(idx_prompt_pair):
            """Helper function for concurrent API calls"""
            orig_idx, prompt = idx_prompt_pair
            import time
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    response_text = _call_judge_api(prompt, attempt=attempt)
                    return orig_idx, prompt, response_text, None
                except Exception as e:
                    print(f"[judge_with_llm_batch] Attempt {attempt+1}/{max_retries} failed for idx {orig_idx}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # tiny sleep to quickly try next key
                    else:
                        return orig_idx, prompt, "", str(e)
        
        # Use ThreadPoolExecutor for concurrent API calls
        new_entries = 0
        with ThreadPoolExecutor(max_workers=_get_max_workers()) as executor:
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
                    with _CACHE_LOCK:
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
#     Call gpt-4o-mini deterministically and cache the output.
#     """
#     global _NEW_ENTRIES_COUNT
#     key = _hash_prompt(prompt)
#     if key in JUDGE_CACHE:
#         return JUDGE_CACHE[key]
#     
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
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