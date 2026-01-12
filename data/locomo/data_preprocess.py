#!/usr/bin/env python3
import json
import re
import random
from pathlib import Path
from datetime import datetime
import pandas as pd

# ===== CONFIG =====
CHUNK_BY_SESSION = True  # If True, each session becomes one chunk (ignores CHUNK_SIZE)
CHUNK_SIZE = 8  # Number of dialogue turns per chunk (only used if CHUNK_BY_SESSION=False)
INPUT_JSON = "locomo10.json"
OUTPUT_DIR = Path("processed")

# Sampling config for QA pairs
USE_ONLY_CURRENT_QAS = True   # If True, only use current QAs (variable count). If False, use balanced sampling.
TARGET_QA_PER_CHUNK = 3       # Fixed number of QAs per chunk for balanced training (only used if USE_ONLY_CURRENT_QAS=False)
MIN_FUTURE_QA = 1             # Always include at least 1 future QA (only used if USE_ONLY_CURRENT_QAS=False)
# RANDOM_SEED = 41              # For reproducibility

# Train/Test/Val split
TRAIN_CONVS = 4
TEST_CONVS = 5
VAL_CONVS = 1

# Post-processing for test data, validation data
MOVE_ALL_QAS_TO_LAST_CHUNK_FOR_TEST = True  # If True, move all QAs to last chunk for test conversations only
MOVE_ALL_QAS_TO_LAST_CHUNK_FOR_VAL = True  # If True, move all QAs to last chunk for val conversations only

# Micro training set for overfitting experiments
CREATE_MICRO_TRAIN = True  # Set to True to create micro_train.{json,parquet}
MICRO_TRAIN_CHUNKS = 10     # Number of chunks for micro training (for learnability testing)

# random.seed(RANDOM_SEED)


# ===== HELPERS =====
def parse_datetime(dt_str):
    """Convert '1:56 pm on 8 May, 2023' → ISO string, else None."""
    if not dt_str:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(dt_str, fmt).isoformat()
        except Exception:
            continue
    return None


_re_dia = re.compile(r"D(?P<sess>\d+):(?P<dia>\d+)")


def parse_dia_id(dia_id):
    """Parse dia_id like 'D8:17' -> (session:int, dia:int)."""
    if not dia_id or not isinstance(dia_id, str):
        return None, None
    m = _re_dia.match(dia_id.strip())
    if not m:
        return None, None
    return int(m.group("sess")), int(m.group("dia"))


def get_session_number_from_key(key):
    """Extract session number from keys like 'session_1' or 'session_10_date_time'."""
    m = re.match(r"session_(\d+)", key)
    return int(m.group(1)) if m else None


def flatten_conversation(conversation: dict):
    """Flatten sessions into one ordered list of turns, sorted by (session_id, dia_id numeric)."""
    turns = []
    session_turns = {}
    session_order = []

    # Collect sessions
    for key in conversation:
        sid = get_session_number_from_key(key)
        if sid is None:
            continue
        if key == f"session_{sid}":
            session_order.append(sid)
            session_turns.setdefault(sid, [])

    if not session_order:
        for key, val in conversation.items():
            if isinstance(val, list) and key.startswith("session_"):
                sid = get_session_number_from_key(key)
                if sid is not None:
                    session_order.append(sid)
                    session_turns.setdefault(sid, [])

    # Fill sessions
    for sid in session_order:
        sess_key = f"session_{sid}"
        sess_time_key = f"session_{sid}_date_time"
        sess_time = conversation.get(sess_time_key)
        items = conversation.get(sess_key, [])
        for idx, t in enumerate(items):
            dia_id = t.get("dia_id")
            _, dia_num = parse_dia_id(dia_id)
            if dia_num is None:
                dia_num = idx + 1
            turn = {
                "session_id": sid,
                "session_time": sess_time,
                **t,  # keep original fields (speaker, dia_id, text, etc.)
            }
            turn["_sort_key"] = (dia_num, idx)
            session_turns[sid].append(turn)

    # Sort and merge
    for sid in sorted(session_turns.keys()):
        session_list_sorted = sorted(session_turns[sid], key=lambda x: x["_sort_key"])
        for t in session_list_sorted:
            t.pop("_sort_key", None)
        turns.extend(session_list_sorted)

    return turns


def find_latest_evidence_session(evidence_list):
    """Return the highest session number from evidence list like ['D1:3','D8:17']"""
    sess_nums = []
    for ev in (evidence_list or []):
        s, _ = parse_dia_id(ev)
        if s is not None:
            sess_nums.append(s)
    return max(sess_nums) if sess_nums else None


def chunk_conversation_by_session(turns, chunk_size=CHUNK_SIZE, chunk_by_session=CHUNK_BY_SESSION):
    """
    Yield (chunk_turns, offset_index, session_id) ensuring chunks don't span sessions.
    
    Args:
        turns: List of conversation turns
        chunk_size: Number of turns per chunk (ignored if chunk_by_session=True)
        chunk_by_session: If True, entire session becomes one chunk
    
    Yields:
        (chunk_turns, offset_index, session_id)
    """
    if not turns:
        return
    
    # Group turns by session
    session_groups = {}
    for turn in turns:
        session_id = turn.get('session_id')
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(turn)
    
    # Process each session independently
    global_offset = 0
    for session_id in sorted(session_groups.keys()):
        session_turns = session_groups[session_id]
        
        if chunk_by_session:
            # Entire session is one chunk
            yield session_turns, global_offset, session_id
            global_offset += len(session_turns)
        else:
            # Chunk within this session by chunk_size
            for i in range(0, len(session_turns), chunk_size):
                chunk_turns = session_turns[i:i + chunk_size]
                yield chunk_turns, global_offset, session_id
                global_offset += len(chunk_turns)

def dia_tuple_from_str(dia_id):
    """Return (session:int, dia:int) or (None,None)."""
    return parse_dia_id(dia_id)

def max_tuple_in_chunk(chunk_turns):
    """Return the max (session, dia) tuple present in chunk_turns."""
    max_t = (0, 0)
    for t in chunk_turns:
        # try parse dia_id first
        s, d = parse_dia_id(t.get("dia_id"))
        if s is None or d is None:
            # fallback: use session_id + an index derived from order
            s = int(t.get("session_id", 0))
            # if dia can't parse, approximate with 0 (or you could use orig index if available)
            d = 0
        cur = (s, d)
        if cur > max_t:
            max_t = cur
    return max_t


def min_tuple_in_chunk(chunk_turns):
    """Return the min (session, dia) tuple present in chunk_turns."""
    min_t = (9999, 9999)
    for t in chunk_turns:
        s, d = parse_dia_id(t.get("dia_id"))
        if s is None or d is None:
            s = int(t.get("session_id", 0))
            d = 0
        cur = (s, d)
        if cur < min_t:
            min_t = cur
    return min_t


def sample_only_current_qas(qa_categorized):
    """
    Return ONLY current QAs (no future, recent, or distant QAs).
    Variable number of QAs per chunk.
    
    Args:
        qa_categorized: Dict with 'current', 'recent', 'distant', 'future' lists
    
    Returns:
        List of all current QAs
    """
    return qa_categorized['current'][:]


def sample_balanced_qas(qa_categorized, target_total=TARGET_QA_PER_CHUNK, min_future=MIN_FUTURE_QA):
    """
    Sample QAs to reach EXACTLY target_total with balanced distribution.
    ALWAYS returns target_total QAs (pads with duplicates if necessary).
    
    Priority:
    1. Include ALL current QAs (most important - new information)
    2. Fill remaining slots uniformly from recent/distant/future (NO preference)
    3. If still short, pad with duplicates
    
    Args:
        qa_categorized: Dict with 'current', 'recent', 'distant', 'future' lists
        target_total: Target number of QAs per chunk (ALWAYS met)
        min_future: Ignored (kept for compatibility)
    
    Returns:
        List of EXACTLY target_total QAs
    """
    current = qa_categorized['current'][:]
    recent = qa_categorized['recent'][:]
    distant = qa_categorized['distant'][:]
    future = qa_categorized['future'][:]
    
    selected_qas = []
    
    # 1. Take all current QAs (priority)
    if len(current) >= target_total:
        # Too many current QAs, sample down to target
        return random.sample(current, target_total)
    
    selected_qas.extend(current)
    remaining = target_total - len(selected_qas)
    
    if remaining <= 0:
        return selected_qas[:target_total]
    
    # 2. Fill remaining slots uniformly by CATEGORY (not by individual QAs)
    # Sample equally from recent/distant/future to avoid bias toward larger categories
    categories = []
    if recent:
        categories.append(('recent', recent))
    if distant:
        categories.append(('distant', distant))
    if future:
        categories.append(('future', future))
    
    if not categories:
        # No other QAs available, pad with current
        if current:
            selected_qas.extend(random.choices(current, k=remaining))
        else:
            # Edge case: no QAs at all
            raise ValueError("No QAs available to sample.")

    else:
        # Sample equally from each available category
        for i in range(remaining):
            # Round-robin through categories to ensure equal representation
            cat_name, cat_qas = categories[i % len(categories)]
            selected_qa = random.choice(cat_qas)
            selected_qas.append(selected_qa)
    
    # Ensure exact count
    return selected_qas[:target_total]


def categorize_qas_by_recency(qa_list, chunk_max_tuple, chunk_min_tuple, only_current=USE_ONLY_CURRENT_QAS):
    """
    Split QAs into categories based on evidence recency:
    - current: Evidence first appears in current chunk
    - recent: Evidence from 1-2 chunks ago
    - distant: Evidence from 3+ chunks ago  
    - future: Evidence not yet seen (should answer "unknown")
    
    Args:
        only_current: If True, only categorize current QAs (others remain empty)
    """
    qas_current = []
    qas_recent = []
    qas_distant = []
    qas_future = []
    
    for qa in qa_list:
        evidence = qa.get("evidence", []) or []
        evidence_tuples = [dia_tuple_from_str(ev) for ev in evidence]
        evidence_tuples = [t for t in evidence_tuples if t[0] is not None and t[1] is not None]
        
        if not evidence_tuples:
            continue
        
        last_evidence = max(evidence_tuples)
        first_evidence = min(evidence_tuples)
        is_adversarial = qa.get("category") == 5
        
        # Current: evidence first appears in this chunk
        if chunk_min_tuple <= last_evidence <= chunk_max_tuple:
            qa_copy = qa.copy()
            qa_copy["qa_type"] = "current"
            if is_adversarial:
                # we need to set answer to unknown for adversarial even if evidence is present
                qa_copy["answer"] = "unknown"
            qas_current.append(qa_copy)
        
        # # If only_current mode, skip the rest
        # elif only_current:
        #     continue
        
        # Future: evidence hasn't appeared yet
        elif last_evidence > chunk_max_tuple:
            # SKIP adversarial (category 5) for future as well
            # We don't want to mix true future questions with adversarial ones
            if is_adversarial:
                continue
            
            qa_copy = qa.copy()
            qa_copy["answer"] = "unknown"
            qa_copy["qa_type"] = "future"
            qas_future.append(qa_copy)
        
        # Past: evidence from earlier chunks
        else:
            # SKIP adversarial (category 5) for recent/distant
            # They were adversarial in the past, but now evidence exists so they're not "unknown" anymore
            if is_adversarial:
                continue
            
            # Calculate dialogue distance in number of dialogue turns
            # Approximate: within same session use dia difference, across sessions use heuristic
            if chunk_max_tuple[0] == last_evidence[0]:
                # Same session - use dialogue number difference
                dia_distance = chunk_max_tuple[1] - last_evidence[1]
            else:
                # Different sessions - estimate distance (assume ~20 dialogues per session on average)
                session_gap = chunk_max_tuple[0] - last_evidence[0]
                dia_distance = session_gap * 20 + chunk_max_tuple[1] - last_evidence[1]
            
            qa_copy = qa.copy()
            # Recent: within last 2-3 chunks (10-15 dialogues with CHUNK_SIZE=5)
            if dia_distance <= 3 * CHUNK_SIZE:
                qa_copy["qa_type"] = "recent"
                qas_recent.append(qa_copy)
            else:
                qa_copy["qa_type"] = "distant"
                qas_distant.append(qa_copy)
    
    return {
        'current': qas_current,
        'recent': qas_recent,
        'distant': qas_distant,
        'future': qas_future
    }


def format_chunk_as_prompt(chunk_turns):
    # TODO:: Now its dummy and will be replaced while training !!, but in general we could do better init
    """
    Minimal prompt string for RLHF dataset compatibility.
    Required as a key but not actually used in our custom generation loop.
    """
    return ""


def process_locomo10(data_path=INPUT_JSON, output_dir=OUTPUT_DIR):
    """Process LoCoMo10 dataset and create train/test/val splits."""
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    
    # Group chunks by conversation first
    all_chunks_by_conv = {}
    
    stats = {
        'total_chunks': 0,
        'total_qas': 0,
        'current_qas': 0,
        'recent_qas': 0,
        'distant_qas': 0,
        'future_qas': 0,
        'duplicate_qas': 0,
        'dummy_qas': 0,
        'partial_chunks': 0,  # Chunks with < CHUNK_SIZE turns
        'chunk_sizes': [],  # Track all chunk sizes for stats
    }

    for conv_idx, conv in enumerate(data):
        sample_id = conv.get("sample_id", f"conv-{conv_idx}")
        qa_list = conv.get("qa", [])
        conversation = conv.get("conversation", {})
        turns = flatten_conversation(conversation)
        
        # Meta information
        speaker_set = set()
        # from first 2 turns to get the 2 speakers
        for t in turns[:2]:
            spk = t.get("speaker")
            if spk:
                speaker_set.add(spk)

        assert len(speaker_set) == 2, "Chunk must contain at most 2 unique speakers."

        conv_chunks = []
        chunk_counter = 1  # Sequential chunk numbering across sessions
        append_to_next = False  # Flag to merge first chunk with next if no QAs

        for chunk_turns, offset, session_id in chunk_conversation_by_session(turns):
            chunk_max_tuple = max_tuple_in_chunk(chunk_turns)
            chunk_min_tuple = min_tuple_in_chunk(chunk_turns)
            
            # Categorize QAs by recency
            qa_categorized = categorize_qas_by_recency(qa_list, chunk_max_tuple, chunk_min_tuple)
            
            # Sample QAs based on configuration
            if USE_ONLY_CURRENT_QAS:
                all_qas = sample_only_current_qas(qa_categorized)
            else:
                all_qas = sample_balanced_qas(qa_categorized)
            
            # Count QAs by type for stats
            qa_type_counts = {'current': 0, 'recent': 0, 'distant': 0, 'future': 0}
            duplicate_count = 0
            dummy_count = 0

            for qa in all_qas:
                if qa.get('is_duplicate', False):
                    duplicate_count += 1
                if qa.get('is_dummy', False):
                    dummy_count += 1
                
                qa_type = qa.get('qa_type', 'current')
                qa_type_counts[qa_type] += 1
            
            # Update stats
            stats['total_chunks'] += 1
            stats['total_qas'] += len(all_qas)
            stats['current_qas'] += qa_type_counts['current']
            stats['recent_qas'] += qa_type_counts['recent']
            stats['distant_qas'] += qa_type_counts['distant']
            stats['future_qas'] += qa_type_counts['future']
            stats['duplicate_qas'] += duplicate_count
            stats['dummy_qas'] += dummy_count
            stats['chunk_sizes'].append(len(chunk_turns))
            if len(chunk_turns) < CHUNK_SIZE:
                stats['partial_chunks'] += 1

            # Extract session_time from the first turn of this chunk
            session_time = chunk_turns[0].get('session_time') if chunk_turns else None

            chunk_data = {
                "sample_id": sample_id,
                "chunk_id": chunk_counter,
                "session_id": session_id,  # Track which session this chunk belongs to
                "session_time": session_time,  # Session timestamp
                "dialogue_num_turns": len(chunk_turns),  # Track actual chunk size
                "num_questions": len(all_qas),  # Track number of QA pairs
                "prompt": format_chunk_as_prompt(chunk_turns),  # Required for RLHF, but wont use it directly
                "turns": chunk_turns,
                "qa_pairs": all_qas,
                "qa_stats": qa_type_counts,
                "speakers": list(speaker_set)  # Include unique speakers in the chunk
            }

            if append_to_next and conv_chunks[-1]['session_id'] == chunk_data['session_id']:
                print(f"⚠️  Merging first chunk of session {chunk_data['session_id']} in conv {sample_id} with next chunk due to no questions.")
                # Merge with previous chunk
                prev_chunk = conv_chunks[-1]
                prev_chunk['turns'].extend(chunk_data['turns'])
                prev_chunk['qa_pairs'].extend(chunk_data['qa_pairs'])
                for qa_type in ['current','recent','distant','future']:
                    prev_chunk['qa_stats'][qa_type] += chunk_data['qa_stats'][qa_type]
                prev_chunk['dialogue_num_turns'] += chunk_data['dialogue_num_turns']
                prev_chunk['num_questions'] += chunk_data['num_questions']
                prev_chunk['prompt'] = format_chunk_as_prompt(prev_chunk['turns'])
                append_to_next = False
            else:
                # Normal case: just append chunk
                conv_chunks.append(chunk_data)
                chunk_counter += 1

            # If I am first chunk of a session, and it has no questions, merge it with next chunk, so save it as temporary and check later
            # check in next iteration, we check counter = 2 because we have already incremented it
            if chunk_data['num_questions'] == 0 and (conv_chunks[-2]["session_id"] != chunk_data['session_id'] if len(conv_chunks) > 1 else True):
                print(f"⚠️  First chunk of session {chunk_data['session_id']} in conv {sample_id} has no questions, will merge with next chunk.")
                append_to_next = True
                continue

            # 🔹 Merge last chunk if it doesnt contain questions
            if (
                len(conv_chunks) > 1
                and chunk_data['num_questions'] == 0
                and conv_chunks[-2]['session_id'] == chunk_data['session_id']
            ):
                prev_chunk = conv_chunks[-2]
                # Merge turns, QAs, update counts
                prev_chunk['turns'].extend(chunk_data['turns'])
                prev_chunk['qa_pairs'].extend(chunk_data['qa_pairs'])
                for qa_type in ['current','recent','distant','future']:
                    prev_chunk['qa_stats'][qa_type] += chunk_data['qa_stats'][qa_type]
                prev_chunk['dialogue_num_turns'] += chunk_data['dialogue_num_turns']
                prev_chunk['num_questions'] += chunk_data['num_questions']
                prev_chunk['prompt'] = format_chunk_as_prompt(prev_chunk['turns'])
                conv_chunks.pop(-1)  # Remove the merged chunk
                chunk_counter -= 1

        
        all_chunks_by_conv[sample_id] = conv_chunks
    
    # Split conversations into train/test/val
    all_conv_ids = sorted(all_chunks_by_conv.keys())
    
    # Shuffle conversations deterministically
    random.shuffle(all_conv_ids)
    
    train_conv_ids = all_conv_ids[:TRAIN_CONVS]
    test_conv_ids = all_conv_ids[TRAIN_CONVS:TRAIN_CONVS + TEST_CONVS]
    val_conv_ids = all_conv_ids[TRAIN_CONVS + TEST_CONVS:TRAIN_CONVS + TEST_CONVS + VAL_CONVS]
    
    # Collect chunks for each split
    train_chunks = []
    test_chunks = []
    val_chunks = []
    
    for conv_id in train_conv_ids:
        train_chunks.extend(all_chunks_by_conv[conv_id])
    
    for conv_id in test_conv_ids:
        test_chunks.extend(all_chunks_by_conv[conv_id])
    
    for conv_id in val_conv_ids:
        val_chunks.extend(all_chunks_by_conv[conv_id])
    
    # ===== POST-PROCESSING: Move all QAs to last chunk for TEST conversations only =====
    if MOVE_ALL_QAS_TO_LAST_CHUNK_FOR_TEST:
        print(f"\n🔄 POST-PROCESSING: Moving all QAs to last chunk for test conversations...")
        
        # Group test_chunks by conversation
        test_chunks_by_conv = {}
        for chunk in test_chunks:
            conv_id = chunk['sample_id']
            if conv_id not in test_chunks_by_conv:
                test_chunks_by_conv[conv_id] = []
            test_chunks_by_conv[conv_id].append(chunk)
        
        # Process each test conversation
        for conv_id, conv_chunks in test_chunks_by_conv.items():
            if not conv_chunks:
                continue
            
            # Collect ALL QA pairs from all chunks
            all_qas_in_conv = []
            for chunk in conv_chunks:
                all_qas_in_conv.extend(chunk['qa_pairs'])
            
            print(f"   📝 Test conv {conv_id}: Collected {len(all_qas_in_conv)} QAs from {len(conv_chunks)} chunks")
            
            # Clear QAs from all chunks except the last one
            for i, chunk in enumerate(conv_chunks):
                if i < len(conv_chunks) - 1:
                    # Not the last chunk - clear QAs
                    chunk['qa_pairs'] = []
                    chunk['num_questions'] = 0
                    chunk['qa_stats'] = {'current': 0, 'recent': 0, 'distant': 0, 'future': 0}
                else:
                    # Last chunk - assign ALL QAs
                    chunk['qa_pairs'] = all_qas_in_conv
                    chunk['num_questions'] = len(all_qas_in_conv)
                    
                    # Recount QA stats for last chunk
                    qa_type_counts = {'current': 0, 'recent': 0, 'distant': 0, 'future': 0}
                    for qa in all_qas_in_conv:
                        qa_type = qa.get('qa_type', 'current')
                        qa_type_counts[qa_type] += 1
                    chunk['qa_stats'] = qa_type_counts
            
            print(f"      ✅ Moved all {len(all_qas_in_conv)} QAs to chunk {len(conv_chunks)} (last chunk)")

    # ===== POST-PROCESSING: Move all QAs to last chunk for VAL conversations only =====
    if MOVE_ALL_QAS_TO_LAST_CHUNK_FOR_VAL:
        print(f"\n🔄 POST-PROCESSING: Moving all QAs to last chunk for val conversations...")
        
        # Group val_chunks by conversation
        val_chunks_by_conv = {}
        for chunk in val_chunks:
            conv_id = chunk['sample_id']
            if conv_id not in val_chunks_by_conv:
                val_chunks_by_conv[conv_id] = []
            val_chunks_by_conv[conv_id].append(chunk)
        
        # Process each val conversation
        for conv_id, conv_chunks in val_chunks_by_conv.items():
            if not conv_chunks:
                continue
            
            # Collect ALL QA pairs from all chunks
            all_qas_in_conv = []
            for chunk in conv_chunks:
                all_qas_in_conv.extend(chunk['qa_pairs'])
            
            print(f"   📝 Val conv {conv_id}: Collected {len(all_qas_in_conv)} QAs from {len(conv_chunks)} chunks")
            
            # Clear QAs from all chunks except the last one
            for i, chunk in enumerate(conv_chunks):
                if i < len(conv_chunks) - 1:
                    # Not the last chunk - clear QAs
                    chunk['qa_pairs'] = []
                    chunk['num_questions'] = 0
                    chunk['qa_stats'] = {'current': 0, 'recent': 0, 'distant': 0, 'future': 0}
                else:
                    # Last chunk - assign ALL QAs
                    chunk['qa_pairs'] = all_qas_in_conv
                    chunk['num_questions'] = len(all_qas_in_conv)
                    
                    # Recount QA stats for last chunk
                    qa_type_counts = {'current': 0, 'recent': 0, 'distant': 0, 'future': 0}
                    for qa in all_qas_in_conv:
                        qa_type = qa.get('qa_type', 'current')
                        qa_type_counts[qa_type] += 1
                    chunk['qa_stats'] = qa_type_counts
            
            print(f"      ✅ Moved all {len(all_qas_in_conv)} QAs to chunk {len(conv_chunks)} (last chunk)")
    
    # Save as both JSON and Parquet
    splits = {
        'train': train_chunks,
        'test': test_chunks,
        'val': val_chunks
    }
    
    # Create micro training set if enabled
    if CREATE_MICRO_TRAIN:
        # Take first MICRO_TRAIN_CHUNKS chunks from each training conversation
        micro_train_chunks = []
        for conv_id in train_conv_ids:
            conv_chunks = all_chunks_by_conv[conv_id]
            micro_train_chunks.extend(conv_chunks[:MICRO_TRAIN_CHUNKS])
        splits['micro_train'] = micro_train_chunks
        print(f"\n🔬 Creating MICRO training set: {len(micro_train_chunks)} chunks (first {MICRO_TRAIN_CHUNKS} from each of {TRAIN_CONVS} conversations)")
    
    split_stats = {}
    
    for split_name, chunks in splits.items():
        # Save as JSON (backup/readable format)
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        # Convert to DataFrame for Parquet
        rows = []
        for chunk in chunks:
            # Flatten the structure for parquet
            row = {
                'sample_id': chunk['sample_id'],
                'chunk_id': chunk['chunk_id'],
                'session_id': chunk['session_id'],  # Which session this chunk belongs to
                'session_time': chunk.get('session_time'),  # Session timestamp
                'prompt': chunk['prompt'],  # RLHF required field
                'turns_json': json.dumps(chunk['turns']),
                'qa_pairs_json': json.dumps(chunk['qa_pairs']),
                'qa_stats_json': json.dumps(chunk['qa_stats']),
                'dialogue_num_turns': chunk['dialogue_num_turns'],  # Actual number of turns (may be < CHUNK_SIZE)
                'num_qas': len(chunk['qa_pairs']),
                'current_qas': chunk['qa_stats']['current'],
                'recent_qas': chunk['qa_stats']['recent'],
                'distant_qas': chunk['qa_stats']['distant'],
                'future_qas': chunk['qa_stats']['future'],
                'speakers': chunk['speakers'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        parquet_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        # Collect stats
        split_stats[split_name] = {
            'conversations': len(set(c['sample_id'] for c in chunks)),
            'chunks': len(chunks),
            'total_qas': sum(len(c['qa_pairs']) for c in chunks),
            'conv_ids': sorted(set(c['sample_id'] for c in chunks))
        }
    
    # Print statistics
    print(f"✅ Processed {stats['total_chunks']} chunks -> {output_dir}/")
    if CHUNK_BY_SESSION:
        print(f"\n📏 Chunk Mode: SESSION-LEVEL (each session = 1 chunk)")
    else:
        print(f"\n📏 Chunk Mode: FIXED-SIZE (chunk_size={CHUNK_SIZE})")
    print(f"   Full chunks ({CHUNK_SIZE} turns): {stats['total_chunks'] - stats['partial_chunks']}")
    print(f"   Partial chunks (< {CHUNK_SIZE} turns): {stats['partial_chunks']}")
    if stats['chunk_sizes']:
        import statistics
        print(f"   Average chunk size: {statistics.mean(stats['chunk_sizes']):.2f} turns")
        print(f"   Min chunk size: {min(stats['chunk_sizes'])} turns")
        print(f"   Max chunk size: {max(stats['chunk_sizes'])} turns")
    
    print(f"\n�📊 Overall QA Distribution:")
    print(f"   Total QAs: {stats['total_qas']}")
    print(f"   Avg QAs per chunk: {stats['total_qas']/stats['total_chunks']:.2f}")
    print(f"   Current QAs: {stats['current_qas']} ({stats['current_qas']/stats['total_qas']*100:.1f}%)")
    print(f"   Recent QAs: {stats['recent_qas']} ({stats['recent_qas']/stats['total_qas']*100:.1f}%)")
    print(f"   Distant QAs: {stats['distant_qas']} ({stats['distant_qas']/stats['total_qas']*100:.1f}%)")
    print(f"   Future QAs: {stats['future_qas']} ({stats['future_qas']/stats['total_qas']*100:.1f}%)")
    
    if stats['duplicate_qas'] > 0 or stats['dummy_qas'] > 0:
        print(f"\n⚠️  Padding Statistics:")
        if stats['duplicate_qas'] > 0:
            print(f"   Duplicate QAs: {stats['duplicate_qas']} ({stats['duplicate_qas']/stats['total_qas']*100:.1f}%)")
        if stats['dummy_qas'] > 0:
            print(f"   Dummy QAs: {stats['dummy_qas']} ({stats['dummy_qas']/stats['total_qas']*100:.1f}%)")
    
    print(f"\n📁 Dataset Splits:")
    split_order = ['train', 'micro_train', 'test', 'val'] if CREATE_MICRO_TRAIN else ['train', 'test', 'val']
    for split_name in split_order:
        if split_name not in split_stats:
            continue
        s = split_stats[split_name]
        
        if split_name == 'micro_train':
            print(f"\n   🔬 {split_name.upper()} (Overfitting Test Set):")
        else:
            print(f"\n   {split_name.upper()}:")
        
        print(f"      Conversations: {s['conversations']} ({', '.join(s['conv_ids'])})")
        print(f"      Chunks: {s['chunks']}")
        print(f"      Total QAs: {s['total_qas']}")
        print(f"      Avg QAs/chunk: {s['total_qas']/s['chunks']:.2f}")
    
    print(f"\n💾 Saved formats:")
    if CREATE_MICRO_TRAIN:
        print(f"   JSON: {output_dir}/{{train,micro_train,test,val}}.json")
        print(f"   Parquet: {output_dir}/{{train,micro_train,test,val}}.parquet")
    else:
        print(f"   JSON: {output_dir}/{{train,test,val}}.json")
        print(f"   Parquet: {output_dir}/{{train,test,val}}.parquet")
    
    print(f"\n💡 Config:")
    print(f"   Chunking: {'SESSION-LEVEL (chunk=session)' if CHUNK_BY_SESSION else f'FIXED-SIZE (chunk_size={CHUNK_SIZE})'}")
    if USE_ONLY_CURRENT_QAS:
        print(f"   QA Sampling: USE_ONLY_CURRENT_QAS=True (variable count, no future/recent/distant QAs)")
    else:
        print(f"   QA Sampling: Balanced mode - Target QAs/chunk={TARGET_QA_PER_CHUNK}, Min future QAs={MIN_FUTURE_QA}")
    
    if CREATE_MICRO_TRAIN:
        print(f"   🔬 Micro train: {MICRO_TRAIN_CHUNKS} chunks (for overfitting/learnability tests)")
    
    return split_stats


if __name__ == "__main__":
    process_locomo10()
