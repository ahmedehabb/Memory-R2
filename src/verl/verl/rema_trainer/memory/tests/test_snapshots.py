"""
Test snapshot functionality for memory manager.
"""
import os
import sys
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager


def test_snapshot_workflow():
    """Test the complete snapshot save/load workflow."""
    
    print("\n" + "="*70)
    print("  SNAPSHOT WORKFLOW TEST")
    print("="*70)
    
    # Set custom cache directory for testing
    os.environ['MEMORY_CACHE_DIR'] = './test_memory_snapshots'
    
    # Simulate processing chunks across epochs
    sample_id = "conv-41"
    
    # ========================================
    # EPOCH 0 - Process chunk 0
    # ========================================
    print("\n" + "-"*70)
    print("EPOCH 0: Processing chunk 0")
    print("-"*70)
    
    memory = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Start with empty memory (chunk 0 at epoch 0)
    snapshot = manager.get_snapshot(sample_id, chunk_id=0, epoch=0)
    if snapshot is None:
        # No snapshot exists yet, use the new memory
        snapshot = memory
    print(f"Starting memory: {len(snapshot.memories)} turns")
    
    # Process chunk 0 - add some memories
    memory.insert(sample_id, 1, "10:00 am", "User", "User's name is John")
    memory.insert(sample_id, 2, "10:05 am", "User", "User likes programming")
    print(f"After processing chunk 0: {len(memory.memories)} turns")
    
    # Cache the state for next chunk (chunk 1 will use this as starting point)
    manager.cache_snapshot(memory, sample_id, chunk_id=1, epoch=0)
    
    # ========================================
    # EPOCH 0 - Process chunk 1
    # ========================================
    print("\n" + "-"*70)
    print("EPOCH 0: Processing chunk 1")
    print("-"*70)
    
    # Start new memory instance
    memory2 = Memory(embedding_method="openai")
    manager2 = MemoryManager(embedding_method="openai")
    
    # Load the snapshot from chunk 0's processing
    snapshot = manager2.get_snapshot(sample_id, chunk_id=1, epoch=0)
    if snapshot is not None:
        memory2.memories = snapshot.memories
        memory2.embedding_matrix = snapshot.embedding_matrix
        memory2.embedding_ids = snapshot.embedding_ids
    
    print(f"Starting memory (from chunk 0): {len(memory2.memories)} turns")
    
    # Process chunk 1 - add more memories
    memory2.insert(sample_id, 3, "10:10 am", "User", "User works at Google")
    memory2.insert(sample_id, 4, "10:15 am", "AI", "Understanding user preferences")
    print(f"After processing chunk 1: {len(memory2.memories)} turns")
    
    # Cache the state for next chunk
    manager2.cache_snapshot(memory2, sample_id, chunk_id=2, epoch=0)
    
    # ========================================
    # EPOCH 1 - Process chunk 0 (uses empty memory)
    # ========================================
    print("\n" + "-"*70)
    print("EPOCH 1: Processing chunk 0")
    print("-"*70)
    
    memory3 = Memory(embedding_method="openai")
    manager3 = MemoryManager(embedding_method="openai")
    
    # Chunk 0 at epoch 1 should start empty (no previous epoch state)
    snapshot = manager3.get_snapshot(sample_id, chunk_id=0, epoch=1)
    if snapshot is None:
        snapshot = memory3
    print(f"Starting memory: {len(snapshot.memories)} turns")
    
    # Process chunk 0 in epoch 1
    memory3.insert(sample_id, 1, "10:00 am", "User", "User's name is John")
    memory3.insert(sample_id, 2, "10:05 am", "User", "User enjoys Python")
    print(f"After processing chunk 0: {len(memory3.memories)} turns")
    
    # Cache for next chunk in epoch 1
    manager3.cache_snapshot(memory3, sample_id, chunk_id=1, epoch=1)
    
    # ========================================
    # EPOCH 1 - Process chunk 1 (uses epoch 1 chunk 0 state)
    # ========================================
    print("\n" + "-"*70)
    print("EPOCH 1: Processing chunk 1")
    print("-"*70)
    
    memory4 = Memory(embedding_method="openai")
    manager4 = MemoryManager(embedding_method="openai")
    
    # Load snapshot from epoch 1, chunk 0
    snapshot = manager4.get_snapshot(sample_id, chunk_id=1, epoch=1)
    if snapshot is not None:
        memory4.memories = snapshot.memories
        memory4.embedding_matrix = snapshot.embedding_matrix
        memory4.embedding_ids = snapshot.embedding_ids
    
    print(f"Starting memory (from epoch 1, chunk 0): {len(memory4.memories)} turns")
    for m in memory4.memories:
        print(f"  - {m['speaker']}: {m['content']}")
    
    # ========================================
    # Verify directory structure
    # ========================================
    print("\n" + "-"*70)
    print("DIRECTORY STRUCTURE")
    print("-"*70)
    
    import subprocess
    result = subprocess.run(
        ["find", "test_memory_snapshots", "-type", "f", "-name", "*.pkl", "-o", "-name", "*.json"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    print("\n" + "="*70)
    print("✓ SNAPSHOT WORKFLOW TEST COMPLETED")
    print("="*70)
    print("\nKey Points:")
    print("  • Each epoch has its own directory: epoch_0/, epoch_1/, etc.")
    print("  • Each conversation has its own subdirectory: conv-41/, conv-42/, etc.")
    print("  • Each chunk creates a snapshot: chunk_0.pkl, chunk_1.pkl, etc.")
    print("  • Chunk N uses the snapshot from chunk N-1 as starting point")
    print("  • Different epochs maintain separate memory states")
    print("  • Embeddings now saved in .pkl files for fast loading!")
    print("  • JSON files also saved for human readability")


if __name__ == "__main__":
    test_snapshot_workflow()
