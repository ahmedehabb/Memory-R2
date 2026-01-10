# judge_llm.py
import os
import json
import hashlib
import atexit
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

# Load main cache first to avoid redundant API calls
JUDGE_CACHE = {}
if os.path.exists(MAIN_CACHE_FILE):
    try:
        with FileLock(LOCK_FILE, timeout=10):
            with open(MAIN_CACHE_FILE, "r") as f:
                JUDGE_CACHE = json.load(f)
        print(f"[cache init] Loaded {len(JUDGE_CACHE)} entries from main cache")
    except Exception as e:
        print(f"[cache init] Could not load main cache: {e}")
        JUDGE_CACHE = {}

# Track new entries added by THIS process only
_NEW_ENTRIES_THIS_PROCESS = set()

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

def merge_to_main_cache():
    """Merge per-process new entries into the main cache safely."""
    global JUDGE_CACHE, _NEW_ENTRIES_THIS_PROCESS
    
    # Only merge if we have new entries from this process
    if not _NEW_ENTRIES_THIS_PROCESS:
        return
    
    # Build dict of only NEW entries to merge
    new_entries = {key: JUDGE_CACHE[key] for key in _NEW_ENTRIES_THIS_PROCESS if key in JUDGE_CACHE}
    if not new_entries:
        return
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with FileLock(LOCK_FILE, timeout=120):
                # Read current main cache
                main_cache = {}
                if os.path.exists(MAIN_CACHE_FILE):
                    try:
                        with open(MAIN_CACHE_FILE, "r") as f:
                            main_cache = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"[merge] Corrupted main cache on attempt {attempt+1}: {e}")
                        # Backup corrupted file
                        backup_file = MAIN_CACHE_FILE + f".corrupted.{os.getpid()}.bak"
                        try:
                            if os.path.exists(MAIN_CACHE_FILE):
                                os.rename(MAIN_CACHE_FILE, backup_file)
                                print(f"[merge] Backed up corrupted cache to {backup_file}")
                        except:
                            pass
                        main_cache = {}
                
                # Merge only new entries from this process
                original_size = len(main_cache)
                main_cache.update(new_entries)
                new_size = len(main_cache)
                added = new_size - original_size
                
                # Atomic write
                tmp_file = MAIN_CACHE_FILE + f".tmp.{os.getpid()}"
                try:
                    with open(tmp_file, "w") as f:
                        json.dump(main_cache, f, indent=2, ensure_ascii=False)
                    
                    # Atomic replace
                    os.replace(tmp_file, MAIN_CACHE_FILE)
                    
                    print(f"[merge success] Merged {added} new entries from process {os.getpid()} (total: {new_size})")
                    
                    # Update local cache and clear tracking
                    JUDGE_CACHE.update(main_cache)
                    _NEW_ENTRIES_THIS_PROCESS.clear()
                    return  # Success!
                    
                except Exception as e:
                    print(f"[merge error] Write failed on attempt {attempt+1}: {e}")
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except:
                            pass
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        print(f"[merge error] Failed after {max_retries} attempts")
                        return
                        
        except Exception as e:
            print(f"[merge error] Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (2 ** attempt))
                continue
            else:
                print(f"[merge error] All {max_retries} attempts failed")
                return

# Register cleanup: merge cache when process exits (best effort, but explicit merge is primary)
atexit.register(merge_to_main_cache)


def judge_with_llm(prompt: str) -> str:
    """
    Call Gemini Flash model deterministically and cache the output.
    Cache is merged to disk by explicit merge_to_main_cache() calls.
    
    Note: Name kept as 'judge_with_llm' for backward compatibility,
    but actually uses Gemini API now.
    """
    global _NEW_ENTRIES_THIS_PROCESS

    key = _hash_prompt(prompt)

    # Return cached result if exists
    if key in JUDGE_CACHE:
        return JUDGE_CACHE[key]

    # Call Gemini API
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

    # Save to cache and track as new entry
    JUDGE_CACHE[key] = result
    _NEW_ENTRIES_THIS_PROCESS.add(key)

    return result


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