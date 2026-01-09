#!/usr/bin/env python3
"""
Analyze the chunked LoCoMo10 dataset for RL training.
Updated to reflect session-aware chunking and fixed QA count per chunk.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

def analyze_chunks(filepath="processed/train.json"):
    """
    Analyze chunks from the new session-aware format.
    Can load from either JSON or Parquet files.
    """
    filepath = Path(filepath)
    
    # Load data based on file type
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.suffix == '.parquet':
        import pandas as pd
        df = pd.read_parquet(filepath)
        # Convert back to dict format
        data = []
        for _, row in df.iterrows():
            chunk = {
                'sample_id': row['sample_id'],
                'chunk_id': row['chunk_id'],
                'session_id': row['session_id'],
                'dialogue_num_turns': row['dialogue_num_turns'],
                'turns': json.loads(row['turns_json']),
                'qa_pairs': json.loads(row['qa_pairs_json']),
                'qa_stats': json.loads(row['qa_stats_json']),
            }
            data.append(chunk)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}. Use .json or .parquet")
    
    print("=" * 80)
    print("LOCOMO10 CHUNKED DATASET ANALYSIS (SESSION-AWARE CHUNKING)")
    print("=" * 80)
    
    # Basic stats
    print(f"\n📦 DATASET OVERVIEW")
    print(f"   Dataset: {filepath.name}")
    print(f"   Total Chunks: {len(data)}")
    conversations = set(c['sample_id'] for c in data)
    print(f"   Unique Conversations: {len(conversations)}")
    print(f"   Avg chunks per conversation: {len(data)/len(conversations):.1f}")
    
    # Session statistics - NEW!
    sessions_per_conv = defaultdict(set)
    chunks_per_session = defaultdict(int)
    for chunk in data:
        sessions_per_conv[chunk['sample_id']].add(chunk['session_id'])
        chunks_per_session[(chunk['sample_id'], chunk['session_id'])] += 1
    
    total_sessions = sum(len(sessions) for sessions in sessions_per_conv.values())
    print(f"\n🔄 SESSION STATISTICS")
    print(f"   Total Sessions: {total_sessions}")
    print(f"   Avg sessions per conversation: {total_sessions/len(conversations):.1f}")
    
    chunks_per_session_counts = Counter(chunks_per_session.values())
    print(f"\n   Chunks per session distribution:")
    for num_chunks, count in sorted(chunks_per_session_counts.items()):
        print(f"      {num_chunks} chunks: {count} sessions")
    
    # Chunk size statistics - UPDATED for session-aware
    chunk_sizes = Counter(c['dialogue_num_turns'] for c in data)
    print(f"\n📏 CHUNK SIZE (conversation turns per chunk)")
    print(f"   Note: Chunks respect session boundaries - last chunk of each session may be partial")
    for size, count in sorted(chunk_sizes.items()):
        pct = count / len(data) * 100
        bar = '█' * int(pct / 2)
        is_partial = " (partial)" if size < 5 else " (full)"
        print(f"   {size} turns: {count:4} chunks ({pct:5.1f}%) {bar}{is_partial}")
    
    # Verify session boundary integrity
    print(f"\n✅ SESSION BOUNDARY VERIFICATION")
    mixed_session_chunks = 0
    for chunk in data:
        session_ids = set(turn.get('session_id') for turn in chunk['turns'])
        if len(session_ids) > 1:
            mixed_session_chunks += 1
    
    if mixed_session_chunks == 0:
        print(f"   ✓ All {len(data)} chunks contain only single-session turns")
        print(f"   ✓ Session boundaries are properly respected!")
    else:
        print(f"   ✗ WARNING: {mixed_session_chunks} chunks mix multiple sessions!")
    
    # QA statistics - CRITICAL: Fixed count per chunk
    total_qas = sum(len(c['qa_pairs']) for c in data)
    qa_counts = [len(c['qa_pairs']) for c in data]
    
    print(f"\n❓ QA PAIR STATISTICS (FIXED COUNT)")
    print(f"   Total QAs: {total_qas}")
    print(f"   Min per chunk: {min(qa_counts)}")
    print(f"   Max per chunk: {max(qa_counts)}")
    print(f"   Avg per chunk: {sum(qa_counts)/len(qa_counts):.2f}")
    print(f"   Median per chunk: {sorted(qa_counts)[len(qa_counts)//2]}")
    
    # Verify all chunks have exactly the target count
    target_count = 6
    exact_target = sum(1 for c in qa_counts if c == target_count)
    pct_exact = exact_target / len(qa_counts) * 100
    print(f"\n   ✅ Chunks with exactly {target_count} QAs: {exact_target}/{len(data)} ({pct_exact:.1f}%)")
    
    if pct_exact < 100:
        print(f"   ⚠️  Distribution of non-{target_count} chunks:")
        other_counts = Counter(c for c in qa_counts if c != target_count)
        for count, freq in sorted(other_counts.items()):
            print(f"      {count} QAs: {freq} chunks")
    
    # Check for padding
    total_duplicates = sum(1 for c in data for qa in c['qa_pairs'] if qa.get('is_duplicate', False))
    total_dummies = sum(1 for c in data for qa in c['qa_pairs'] if qa.get('is_dummy', False))
    
    if total_duplicates > 0 or total_dummies > 0:
        print(f"\n   ⚠️  Padding Used:")
        if total_duplicates > 0:
            chunks_with_dups = sum(1 for c in data if any(qa.get('is_duplicate', False) for qa in c['qa_pairs']))
            print(f"      Duplicate QAs: {total_duplicates} (in {chunks_with_dups} chunks)")
        if total_dummies > 0:
            chunks_with_dummies = sum(1 for c in data if any(qa.get('is_dummy', False) for qa in c['qa_pairs']))
            print(f"      Dummy QAs: {total_dummies} (in {chunks_with_dummies} chunks)")
    else:
        print(f"   ✅ No padding needed - all chunks filled naturally!")
    
    # QA type distribution - Critical for memory training
    qa_types = Counter()
    qa_categories = Counter()
    answers_unknown = 0
    
    for chunk in data:
        for qa in chunk['qa_pairs']:
            qa_type = qa.get('qa_type', 'unknown')
            qa_types[qa_type] += 1
            qa_categories[qa.get('category', 'N/A')] += 1
            if qa.get('answer') == 'unknown':
                answers_unknown += 1
    
    print(f"\n🏷️  QA TYPE DISTRIBUTION (Memory Training)")
    print(f"   {'Type':<12} {'Count':>6} {'Percentage':>12} {'Purpose':<35}")
    print(f"   {'-'*12} {'-'*6} {'-'*12} {'-'*35}")
    
    type_purposes = {
        'current': 'Answer from current chunk',
        'recent': 'Short-term memory (1-2 chunks ago)',
        'distant': 'Long-term memory (3+ chunks ago)',
        'future': 'Should answer "unknown"'
    }
    
    for qa_type in ['current', 'recent', 'distant', 'future']:
        count = qa_types.get(qa_type, 0)
        pct = count / total_qas * 100 if total_qas > 0 else 0
        purpose = type_purposes.get(qa_type, '')
        print(f"   {qa_type:<12} {count:>6} {pct:>11.1f}% {purpose:<35}")
    
    # Show any unknown types
    for qa_type, count in qa_types.items():
        if qa_type not in type_purposes:
            pct = count / total_qas * 100
            print(f"   {qa_type:<12} {count:>6} {pct:>11.1f}% {'Unknown type':<35}")
    
    print(f"\n📂 QA CATEGORY DISTRIBUTION (Original Dataset)")
    cat_descriptions = {
        1: "Factual recall",
        2: "Temporal reasoning", 
        3: "Multi-hop inference",
        4: "Conversation understanding",
        5: "Adversarial (unknown)"
    }
    for cat in sorted(qa_categories.keys()):
        count = qa_categories[cat]
        pct = count / total_qas * 100
        desc = cat_descriptions.get(cat, "Unknown")
        print(f"   Category {cat}: {count:5} ({pct:5.1f}%) - {desc}")
    
    print(f"\n🚫 UNKNOWN ANSWER HANDLING")
    print(f"   Total 'unknown' answers: {answers_unknown} ({answers_unknown/total_qas*100:.1f}%)")
    print(f"   (Includes future QAs + adversarial category 5)")
    
    # Per-chunk QA type balance
    print(f"\n📊 PER-CHUNK QA TYPE BALANCE")
    type_stats = {t: [] for t in ['current', 'recent', 'distant', 'future']}
    
    for chunk in data:
        stats = chunk['qa_stats']
        for qa_type in type_stats:
            type_stats[qa_type].append(stats.get(qa_type, 0))
    
    print(f"   {'Type':<12} {'Min':>5} {'Max':>5} {'Avg':>6} {'Median':>7}")
    print(f"   {'-'*12} {'-'*5} {'-'*5} {'-'*6} {'-'*7}")
    for qa_type in ['current', 'recent', 'distant', 'future']:
        counts = type_stats[qa_type]
        min_val = min(counts)
        max_val = max(counts)
        avg_val = sum(counts) / len(counts)
        med_val = sorted(counts)[len(counts)//2]
        print(f"   {qa_type:<12} {min_val:>5} {max_val:>5} {avg_val:>6.2f} {med_val:>7}")
    
    # Chunk size distribution
    chunk_sizes = Counter(len(c['turns']) for c in data)
    total_turns = sum(len(c['turns']) for c in data)
    print(f"\n💬 CONVERSATION TURNS")
    print(f"   Total turns across all chunks: {total_turns}")
    print(f"   Avg turns per chunk: {total_turns/len(data):.2f}")
    
    # Sample conversation analysis
    print(f"\n" + "=" * 80)
    print("SAMPLE CONVERSATION ANALYSIS (Detailed)")
    print("=" * 80)
    
    # Pick first conversation as example
    example_conv = sorted(conversations)[0]
    conv_chunks = [c for c in data if c['sample_id'] == example_conv]
    conv_chunks.sort(key=lambda x: x['chunk_id'])
    
    print(f"\nConversation: {example_conv}")
    print(f"Total chunks: {len(conv_chunks)}")
    
    # Session breakdown for this conversation
    sessions_in_conv = sorted(set(c['session_id'] for c in conv_chunks))
    print(f"Sessions: {len(sessions_in_conv)} ({', '.join(f'S{s}' for s in sessions_in_conv)})")
    
    chunks_by_session = defaultdict(list)
    for chunk in conv_chunks:
        chunks_by_session[chunk['session_id']].append(chunk)
    
    print(f"\n   Session breakdown:")
    for sess_id in sessions_in_conv:
        sess_chunks = chunks_by_session[sess_id]
        sess_turns = sum(c['dialogue_num_turns'] for c in sess_chunks)
        sess_qas = sum(len(c['qa_pairs']) for c in sess_chunks)
        print(f"      Session {sess_id}: {len(sess_chunks)} chunks, {sess_turns} turns, {sess_qas} QAs")
    
    print(f"\nTotal turns: {sum(c['dialogue_num_turns'] for c in conv_chunks)}")
    print(f"Total QAs: {sum(len(c['qa_pairs']) for c in conv_chunks)}")
    
    print(f"\nFirst 15 chunks breakdown:")
    print(f"{'#':<4} {'Chunk':<7} {'Sess':<5} {'Turns':>6} {'Turn Range':<15} {'QAs':>5} {'Cur':>4} {'Rec':>4} {'Dis':>4} {'Fut':>4}")
    print("-" * 90)
    
    for i, chunk in enumerate(conv_chunks[:15]):
        stats = chunk['qa_stats']
        if chunk['turns']:
            turn_range = f"{chunk['turns'][0].get('dia_id', 'N/A')}-{chunk['turns'][-1].get('dia_id', 'N/A')}"
        else:
            turn_range = "empty"
        total_qas = len(chunk['qa_pairs'])
        print(f"{i+1:<4} {chunk['chunk_id']:<7} S{chunk['session_id']:<4} {chunk['dialogue_num_turns']:>6} {turn_range:<15} {total_qas:>5} "
              f"{stats.get('current', 0):>4} {stats.get('recent', 0):>4} "
              f"{stats.get('distant', 0):>4} {stats.get('future', 0):>4}")
    
    # Show progression of QA types over conversation
    print(f"\n📈 QA TYPE PROGRESSION (shows memory accumulation)")
    print(f"   Early chunks (1-10): More current & future, less past memory")
    print(f"   Mid chunks (11-40): Balanced mix, memory retrieval increases")
    print(f"   Late chunks (41+): More memory retrieval from distant past")
    
    if len(conv_chunks) >= 40:
        early = conv_chunks[:10]
        mid = conv_chunks[20:30]
        late = conv_chunks[40:50] if len(conv_chunks) >= 50 else conv_chunks[-10:]
        
        for label, chunks in [("Early (1-10)", early), ("Mid (21-30)", mid), ("Late (41-50)", late)]:
            cur = sum(c['qa_stats'].get('current', 0) for c in chunks)
            rec = sum(c['qa_stats'].get('recent', 0) for c in chunks)
            dis = sum(c['qa_stats'].get('distant', 0) for c in chunks)
            fut = sum(c['qa_stats'].get('future', 0) for c in chunks)
            total = cur + rec + dis + fut
            
            print(f"\n   {label}:")
            print(f"      Current: {cur:3} ({cur/total*100:5.1f}%)  Recent: {rec:3} ({rec/total*100:5.1f}%)")
            print(f"      Distant: {dis:3} ({dis/total*100:5.1f}%)  Future: {fut:3} ({fut/total*100:5.1f}%)")
    
    # Example QAs from middle chunk
    if len(conv_chunks) >= 10:
        print(f"\n" + "=" * 80)
        print("EXAMPLE QAs FROM CHUNK 10 (Middle of conversation)")
        print("=" * 80)
        
        chunk10 = conv_chunks[9]
        print(f"\nTurns: {chunk10['turns'][0]['dia_id']} to {chunk10['turns'][-1]['dia_id']}")
        print(f"Total QAs: {len(chunk10['qa_pairs'])}\n")
        
        print(f"{'Type':<10} {'Question':<50} {'Answer':<25}")
        print("-" * 85)
        
        for qa in chunk10['qa_pairs']:
            qa_type = qa.get('qa_type', 'N/A')
            question = qa['question'][:47] + "..." if len(qa['question']) > 50 else qa['question']
            answer = str(qa['answer'])[:22] + "..." if len(str(qa['answer'])) > 25 else str(qa['answer'])
            print(f"{qa_type:<10} {question:<50} {answer:<25}")
        
    print("\n" + "=" * 80)

def analyze_all_splits(base_dir="processed"):
    """Analyze all train/test/val splits."""
    base_path = Path(base_dir)
    
    print("=" * 80)
    print("ANALYZING ALL DATASET SPLITS")
    print("=" * 80)
    
    for split in ['train', 'micro_train', 'test', 'val']:
        json_path = base_path / f"{split}.json"
        if json_path.exists():
            print(f"\n{'='*80}")
            if split == 'micro_train':
                print(f"SPLIT: {split.upper()} (OVERFITTING TEST SET)")
            else:
                print(f"SPLIT: {split.upper()}")
            print(f"{'='*80}")
            analyze_chunks(str(json_path))
        else:
            if split != 'micro_train':  # Don't warn if micro_train doesn't exist (optional)
                print(f"\n⚠️  {split}.json not found in {base_dir}/")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific file
        analyze_chunks(sys.argv[1])
    else:
        # Analyze all splits
        analyze_all_splits()
