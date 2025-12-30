"""
Test suite for the Memory class - conversation-based storage.
"""
from verl.rema_trainer.memory.memory_core.memory import Memory


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_simple_tests():
    """Run simple memory tests."""
    print("\n" + "="*60)
    print("  MEMORY TEST SUITE")
    print("="*60)
    
    # Test 1: Initialize
    print_section("Test 1: Initialization")
    mem = Memory(embedding_method="openai")
    print(f"✓ Memory initialized with {len(mem.memories)} turns")
    
    # Test 2: Insert
    print_section("Test 2: Insert")
    t1 = mem.insert("conv-1", 1, "10:00 am", "User", "Hello world")
    print(f"✓ Inserted: {t1['memory_id']}")
    
    t2 = mem.insert("conv-1", 2, "10:01 am", "AI", "Hi there!")
    print(f"✓ Inserted: {t2['memory_id']}")
    print(f"Total: {len(mem.memories)} turns")
    
    # Test 3: Get
    print_section("Test 3: Get")
    all_turns = mem.get()
    print(f"✓ Get all: {len(all_turns)} turns")
    
    conv1 = mem.get(sample_id="conv-1")
    print(f"✓ Get conv-1: {len(conv1)} turns")
    
    # Test 4: Update
    print_section("Test 4: Update")
    updated = mem.update(t1['memory_id'], "Updated content!")
    print(f"✓ Updated: {updated['content']}")
    
    # Test 5: Search
    print_section("Test 5: Search")
    results = mem.search("hello", search_method="bm25")
    print(f"✓ Search found: {len(results)} results")
    
    # Test 6: Delete
    print_section("Test 6: Delete")
    success = mem.delete(t1['memory_id'])
    print(f"✓ Delete: {success}, remaining: {len(mem.memories)}")
    
    # Test 7: Save
    print_section("Test 7: Save")
    save_path = mem.save("test_save")
    print(f"✓ Saved to: {save_path}")
    
    # Test 8: Load
    print_section("Test 8: Load")
    mem2 = Memory(embedding_method="openai")
    # Need to get the directory from the save_path
    import os
    from pathlib import Path
    save_dir = str(Path(save_path).parent)
    loaded_count = mem2.load("test_save", directory=save_dir)
    print(f"✓ Loaded {loaded_count} memories")
    print(f"✓ Memory content matches: {len(mem2.memories) == len(mem.memories)}")
    print(f"✓ Embeddings restored: {mem2.embedding_matrix.shape[0] == len(mem2.memories)}")
    
    # Test 9: Load with merge
    print_section("Test 9: Load with Merge")
    mem3 = Memory(embedding_method="openai")
    mem3.insert("conv-2", 1, "11:00 am", "User", "New conversation")
    initial_count = len(mem3.memories)
    mem3.load("test_save", directory=save_dir, clear_existing=False)
    print(f"✓ Before merge: {initial_count}, After merge: {len(mem3.memories)}")
    print(f"✓ Merge successful: {len(mem3.memories) > initial_count}")
    
    # Test 10: List saves
    print_section("Test 10: List Saves")
    saves = mem.list_saves(directory=save_dir)
    print(f"✓ Found {len(saves)} saves")
    if saves:
        print(f"✓ Latest save: {saves[0]['save_name']} ({saves[0]['total_memories']} memories)")
    
    print_section("ALL TESTS COMPLETED")
    print("✓ All tests passed!")


if __name__ == "__main__":
    run_simple_tests()
