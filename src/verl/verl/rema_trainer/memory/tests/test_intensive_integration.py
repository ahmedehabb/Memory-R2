"""
INTENSIVE INTEGRATION TEST SUITE FOR MEMORY SYSTEM
===================================================
Tests ALL functionalities with REAL examples:
- Insertion (single, bulk, duplicates)
- Search (text-embedding, BM25, filters)
- Retrieval (get, filters by sample_id/speaker)
- Update (content updates, embedding regeneration)
- Delete (single, multiple, cascade effects)
- Cache (hit/miss, persistence, cost tracking)
- Persistence (save, load, merge, snapshots)
- Edge cases and stress testing

Author: Test Suite
Date: December 29, 2025
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_core.memory import Memory
from memory_core.memory_manager import MemoryManager
from memory_core.embedding_cache import EmbeddingCache


# ===== UTILITY FUNCTIONS =====

def print_banner(text: str):
    """Print a formatted banner."""
    print(f"\n{'='*80}")
    print(f"  {text.upper()}")
    print(f"{'='*80}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'-'*80}")
    print(f"  {text}")
    print(f"{'-'*80}")


def print_success(message: str):
    """Print success message."""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"✗ ERROR: {message}")


def print_info(message: str):
    """Print info message."""
    print(f"ℹ {message}")


def assert_test(condition: bool, message: str):
    """Assert a test condition."""
    if condition:
        print_success(message)
    else:
        print_error(message)
        raise AssertionError(f"Test failed: {message}")


# ===== REAL SAMPLE DATA =====

REAL_CONVERSATIONS = {
    "conv-python-learning": [
        {
            "sample_id": "conv-python-learning",
            "session_id": 1,
            "session_time": "11:00 AM on December 1, 2024",
            "speaker": "Student",
            "content": "I want to learn Python programming. Where should I start?"
        },
        {
            "sample_id": "conv-python-learning",
            "session_id": 2,
            "session_time": "11:02 AM on December 1, 2024",
            "speaker": "Teacher",
            "content": "Great choice! Start with Python basics: variables, data types, loops, and functions. I recommend the official Python tutorial at python.org."
        },
        {
            "sample_id": "conv-python-learning",
            "session_id": 3,
            "session_time": "11:05 AM on December 1, 2024",
            "speaker": "Student",
            "content": "What about data structures? Are lists and dictionaries important?"
        },
        {
            "sample_id": "conv-python-learning",
            "session_id": 4,
            "session_time": "11:07 AM on December 1, 2024",
            "speaker": "Teacher",
            "content": "Absolutely! Lists and dictionaries are fundamental. Lists store ordered collections, while dictionaries store key-value pairs. Master these first."
        },
    ],
    "conv-ml-discussion": [
        {
            "sample_id": "conv-ml-discussion",
            "session_id": 1,
            "session_time": "2:00 PM on December 2, 2024",
            "speaker": "Researcher",
            "content": "Machine learning has revolutionized AI. Neural networks can now perform tasks like image recognition with human-level accuracy."
        },
        {
            "sample_id": "conv-ml-discussion",
            "session_id": 2,
            "session_time": "2:03 PM on December 2, 2024",
            "speaker": "Student",
            "content": "How do neural networks actually learn? Is it just adjusting weights?"
        },
        {
            "sample_id": "conv-ml-discussion",
            "session_id": 3,
            "session_time": "2:05 PM on December 2, 2024",
            "speaker": "Researcher",
            "content": "Yes! Backpropagation adjusts weights using gradient descent to minimize loss. The network learns patterns by iteratively updating parameters."
        },
    ],
    "conv-cooking-tips": [
        {
            "sample_id": "conv-cooking-tips",
            "session_id": 1,
            "session_time": "6:00 PM on December 3, 2024",
            "speaker": "Chef",
            "content": "The secret to great pasta is using salted boiling water and not overcooking. Al dente is perfect!"
        },
        {
            "sample_id": "conv-cooking-tips",
            "session_id": 2,
            "session_time": "6:02 PM on December 3, 2024",
            "speaker": "Home Cook",
            "content": "What about the sauce? Should I add pasta water to it?"
        },
        {
            "sample_id": "conv-cooking-tips",
            "session_id": 3,
            "session_time": "6:04 PM on December 3, 2024",
            "speaker": "Chef",
            "content": "Excellent question! Yes, pasta water contains starch that helps bind the sauce. Add it gradually while tossing."
        },
    ],
    "conv-space-exploration": [
        {
            "sample_id": "conv-space-exploration",
            "session_id": 1,
            "session_time": "10:00 AM on December 4, 2024",
            "speaker": "Scientist",
            "content": "The James Webb Space Telescope has revealed galaxies from 13 billion years ago, showing us the universe's infancy."
        },
        {
            "sample_id": "conv-space-exploration",
            "session_id": 2,
            "session_time": "10:03 AM on December 4, 2024",
            "speaker": "Student",
            "content": "How does it see so far back in time? Is it because light takes time to travel?"
        },
        {
            "sample_id": "conv-space-exploration",
            "session_id": 3,
            "session_time": "10:05 AM on December 4, 2024",
            "speaker": "Scientist",
            "content": "Exactly! Light from distant galaxies takes billions of years to reach us. When we observe them, we're seeing the past."
        },
    ],
}


# ===== TEST 1: BASIC INSERTION =====

def test_basic_insertion():
    """Test basic insertion operations."""
    print_banner("Test 1: Basic Insertion Operations")
    
    mem = Memory(embedding_method="openai", enable_cache=True)
    
    print_section("1.1: Insert Single Memory")
    result = mem.insert(
        sample_id="conv-test",
        session_id=1,
        session_time="12:00 PM",
        speaker="User",
        content="This is a test message"
    )
    assert_test(result is not None, "Memory inserted successfully")
    assert_test("memory_id" in result, "Memory ID generated")
    assert_test(len(mem.memories) == 1, "Memory count is 1")
    print_info(f"Inserted memory ID: {result['memory_id']}")
    
    print_section("1.2: Insert Multiple Memories")
    for i, conv_data in enumerate(REAL_CONVERSATIONS["conv-python-learning"][:2]):
        result = mem.insert(**conv_data)
        print_success(f"Inserted memory {i+1}: {result['memory_id']}")
    
    assert_test(len(mem.memories) == 3, "Total memories: 3")
    
    print_section("1.3: Duplicate Detection")
    # Try to insert the same content again
    initial_count = len(mem.memories)
    duplicate = mem.insert(
        sample_id="conv-test",
        session_id=1,
        session_time="12:00 PM",
        speaker="User",
        content="This is a test message"
    )
    assert_test(len(mem.memories) == initial_count, "Duplicate not inserted")
    print_success("Duplicate detection working correctly")
    
    print_section("1.4: Bulk Insertion")
    initial_count = len(mem.memories)
    for conv_name, conv_data in REAL_CONVERSATIONS.items():
        for turn in conv_data:
            mem.insert(**turn)
    
    final_count = len(mem.memories)
    assert_test(final_count > initial_count, f"Bulk insertion: {final_count - initial_count} new memories")
    print_success(f"Total memories in system: {final_count}")
    
    return mem


# ===== TEST 2: SEARCH FUNCTIONALITY =====

def test_search_functionality(mem: Memory):
    """Test all search methods comprehensively."""
    print_banner("Test 2: Search Functionality")
    
    print_section("2.1: Text Embedding Search - Python")
    results = mem.search(
        query="Python programming and data structures",
        search_method="text-embedding",
        top_k=5
    )
    assert_test(len(results) > 0, "Found results for Python query")
    print_info(f"Found {len(results)} results")
    for i, (memory, score) in enumerate(results[:3]):
        print(f"  {i+1}. Score: {score:.4f} | Speaker: {memory['speaker']} | Content: {memory['content'][:60]}...")
    
    print_section("2.2: Text Embedding Search - Machine Learning")
    results = mem.search(
        query="neural networks and deep learning algorithms",
        search_method="text-embedding",
        top_k=5
    )
    assert_test(len(results) > 0, "Found results for ML query")
    print_info(f"Found {len(results)} results")
    for i, (memory, score) in enumerate(results[:3]):
        print(f"  {i+1}. Score: {score:.4f} | Sample: {memory['sample_id']} | Content: {memory['content'][:60]}...")
    
    print_section("2.3: BM25 Search")
    results = mem.search(
        query="pasta water sauce cooking",
        search_method="bm25",
        top_k=5
    )
    assert_test(len(results) > 0, "Found results with BM25")
    print_info(f"Found {len(results)} results")
    for i, (memory, score) in enumerate(results[:3]):
        print(f"  {i+1}. Score: {score:.4f} | Speaker: {memory['speaker']} | Content: {memory['content'][:60]}...")
    
    print_section("2.4: Search with sample_id Filter")
    results = mem.search(
        query="learning",
        sample_id="conv-python-learning",
        search_method="text-embedding"
    )
    assert_test(len(results) > 0, "Found results with sample_id filter")
    assert_test(all(m[0]["sample_id"] == "conv-python-learning" for m in results), 
                "All results from correct conversation")
    print_success(f"Filter working: {len(results)} results from conv-python-learning")
    
    print_section("2.5: Search with Speaker Filter")
    results = mem.search(
        query="teaching",
        speaker="Teacher",
        search_method="text-embedding"
    )
    assert_test(all(m[0]["speaker"] == "Teacher" for m in results), 
                "All results from Teacher speaker")
    print_success(f"Speaker filter working: {len(results)} results from Teacher")
    
    print_section("2.6: Search with Combined Filters")
    results = mem.search(
        query="Python",
        sample_id="conv-python-learning",
        speaker="Student",
        search_method="text-embedding"
    )
    print_info(f"Found {len(results)} results with combined filters")
    for memory, score in results:
        assert_test(memory["sample_id"] == "conv-python-learning", "Correct sample_id")
        assert_test(memory["speaker"] == "Student", "Correct speaker")
    
    print_section("2.7: Search with Minimum Score")
    results = mem.search(
        query="Python programming",
        search_method="text-embedding",
        min_score=0.3
    )
    assert_test(all(score >= 0.3 for _, score in results), 
                "All results above minimum score")
    print_success(f"Min score filter: {len(results)} results with score >= 0.3")
    
    print_section("2.8: Empty Query Handling")
    results = mem.search(query="", search_method="text-embedding")
    assert_test(len(results) == 0, "Empty query returns no results")
    print_success("Empty query handled correctly")


# ===== TEST 3: RETRIEVAL OPERATIONS =====

def test_retrieval_operations(mem: Memory):
    """Test get operations with various filters."""
    print_banner("Test 3: Retrieval Operations")
    
    print_section("3.1: Get All Memories")
    all_memories = mem.get()
    assert_test(len(all_memories) > 0, f"Retrieved all memories: {len(all_memories)}")
    print_info(f"Total memories: {len(all_memories)}")
    
    print_section("3.2: Get by sample_id")
    python_memories = mem.get(sample_id="conv-python-learning")
    assert_test(len(python_memories) > 0, "Retrieved Python conversation memories")
    assert_test(all(m["sample_id"] == "conv-python-learning" for m in python_memories),
                "All memories from correct conversation")
    print_success(f"Found {len(python_memories)} memories for conv-python-learning")
    
    print_section("3.3: Get by Speaker")
    teacher_memories = mem.get(speaker="Teacher")
    assert_test(len(teacher_memories) > 0, "Retrieved Teacher memories")
    assert_test(all(m["speaker"] == "Teacher" for m in teacher_memories),
                "All memories from Teacher")
    print_success(f"Found {len(teacher_memories)} memories from Teacher")
    
    print_section("3.4: Get with Combined Filters")
    filtered = mem.get(sample_id="conv-python-learning", speaker="Student")
    assert_test(all(m["sample_id"] == "conv-python-learning" and m["speaker"] == "Student" 
                   for m in filtered),
                "Combined filter working correctly")
    print_success(f"Found {len(filtered)} memories with combined filters")
    
    print_section("3.5: Get Non-existent sample_id")
    empty = mem.get(sample_id="conv-nonexistent")
    assert_test(len(empty) == 0, "Non-existent sample_id returns empty list")
    print_success("Non-existent sample_id handled correctly")


# ===== TEST 4: UPDATE OPERATIONS =====

def test_update_operations(mem: Memory):
    """Test update operations and embedding regeneration."""
    print_banner("Test 4: Update Operations")
    
    print_section("4.1: Update Memory Content")
    # Get a memory to update
    memories = mem.get(sample_id="conv-python-learning")
    if not memories:
        print_error("No memories to update")
        return
    
    target_memory = memories[0]
    original_content = target_memory["content"]
    memory_id = target_memory["memory_id"]
    
    print_info(f"Original content: {original_content[:60]}...")
    
    # Get original embedding
    if memory_id in mem.embedding_ids:
        idx = mem.embedding_ids.index(memory_id)
        original_embedding = mem.embedding_matrix[idx].copy()
    
    # Update the memory
    new_content = "This is completely new and different content about quantum physics and relativity"
    updated = mem.update(memory_id, new_content)
    
    assert_test(updated is not None, "Memory updated successfully")
    assert_test(updated["content"] == new_content, "Content updated correctly")
    print_success(f"Updated content: {new_content[:60]}...")
    
    print_section("4.2: Verify Embedding Regeneration")
    # Check that embedding was regenerated
    if memory_id in mem.embedding_ids:
        idx = mem.embedding_ids.index(memory_id)
        new_embedding = mem.embedding_matrix[idx]
        
        # Embeddings should be different for completely different content
        similarity = np.dot(original_embedding, new_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(new_embedding)
        )
        assert_test(similarity < 0.9, f"Embedding changed (similarity: {similarity:.4f})")
        print_success("Embedding regenerated correctly")
    
    print_section("4.3: Update Non-existent Memory")
    result = mem.update("nonexistent-id", "New content")
    assert_test(result is None, "Non-existent memory returns None")
    print_success("Non-existent memory handled correctly")
    
    print_section("4.4: Multiple Updates")
    # Test updating the same memory multiple times
    for i in range(3):
        new_content = f"Update iteration {i+1}: Testing repeated updates"
        updated = mem.update(memory_id, new_content)
        assert_test(updated["content"] == new_content, f"Update {i+1} successful")
    print_success("Multiple updates working correctly")


# ===== TEST 5: DELETE OPERATIONS =====

def test_delete_operations(mem: Memory):
    """Test delete operations and cascade effects."""
    print_banner("Test 5: Delete Operations")
    
    initial_count = len(mem.memories)
    initial_embedding_count = mem.embedding_matrix.shape[0]
    
    print_section("5.1: Delete Single Memory")
    # Get a memory to delete
    memories = mem.get(sample_id="conv-cooking-tips")
    if not memories:
        print_error("No memories to delete")
        return
    
    target_memory = memories[0]
    memory_id = target_memory["memory_id"]
    print_info(f"Deleting memory: {memory_id}")
    
    success = mem.delete(memory_id)
    assert_test(success, "Memory deleted successfully")
    assert_test(len(mem.memories) == initial_count - 1, "Memory count decreased")
    assert_test(mem.embedding_matrix.shape[0] == initial_embedding_count - 1, 
                "Embedding count decreased")
    print_success("Memory and embedding deleted")
    
    print_section("5.2: Verify Deletion")
    # Try to retrieve the deleted memory
    all_ids = [m["memory_id"] for m in mem.memories]
    assert_test(memory_id not in all_ids, "Memory ID not in memory list")
    assert_test(memory_id not in mem.embedding_ids, "Memory ID not in embedding list")
    print_success("Deletion verified")
    
    print_section("5.3: Delete Non-existent Memory")
    success = mem.delete("nonexistent-id")
    assert_test(not success, "Non-existent memory returns False")
    print_success("Non-existent memory handled correctly")
    
    print_section("5.4: Delete Multiple Memories")
    # Delete all memories from a specific conversation
    to_delete = mem.get(sample_id="conv-space-exploration")
    delete_count = len(to_delete)
    print_info(f"Deleting {delete_count} memories from conv-space-exploration")
    
    for memory in to_delete:
        success = mem.delete(memory["memory_id"])
        assert_test(success, f"Deleted {memory['memory_id']}")
    
    # Verify all deleted
    remaining = mem.get(sample_id="conv-space-exploration")
    assert_test(len(remaining) == 0, "All memories from conversation deleted")
    print_success(f"Successfully deleted {delete_count} memories")
    
    print_section("5.5: Verify Embedding Matrix Consistency")
    assert_test(len(mem.memories) == mem.embedding_matrix.shape[0],
                "Memory count matches embedding matrix size")
    assert_test(len(mem.memories) == len(mem.embedding_ids),
                "Memory count matches embedding ID list size")
    print_success("Embedding matrix consistent after deletions")


# ===== TEST 6: CACHE FUNCTIONALITY =====

def test_cache_functionality():
    """Test embedding cache hit/miss, persistence, and cost tracking."""
    print_banner("Test 6: Cache Functionality")
    
    # Create a fresh cache directory for testing
    cache_dir = Path(__file__).parent / "test_cache_intensive"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    
    print_section("6.1: Cache Initialization")
    mem = Memory(embedding_method="openai", enable_cache=True, cache_dir=str(cache_dir))
    assert_test(mem.cache is not None, "Cache initialized")
    # Directory is created lazily when first cache entry is added
    print_success(f"Cache configured for directory: {cache_dir}")
    
    print_section("6.2: First Embedding (Cache Miss)")
    test_text = "This is a unique test text for cache testing"
    initial_misses = mem.cache.stats["misses"]
    embedding1 = mem._get_embedding(test_text)
    
    assert_test(mem.cache.stats["misses"] == initial_misses + 1, "Cache miss recorded")
    assert_test(embedding1.shape[0] == 1536, "Correct embedding dimension")
    print_success("First embedding generated (cache miss)")
    print_info(f"Cache stats: {mem.cache.stats}")
    
    print_section("6.3: Second Embedding (Cache Hit)")
    initial_hits = mem.cache.stats["hits"]
    embedding2 = mem._get_embedding(test_text)
    
    assert_test(mem.cache.stats["hits"] == initial_hits + 1, "Cache hit recorded")
    assert_test(np.allclose(embedding1, embedding2), "Embeddings match from cache")
    print_success("Second embedding retrieved from cache (cache hit)")
    print_info(f"Cache stats: {mem.cache.stats}")
    
    print_section("6.4: Cache Persistence")
    # Create new memory instance with same cache
    mem2 = Memory(embedding_method="openai", enable_cache=True, cache_dir=str(cache_dir))
    embedding3 = mem2._get_embedding(test_text)
    
    assert_test(np.allclose(embedding1, embedding3), "Cache persists across instances")
    print_success("Cache persisted correctly")
    
    print_section("6.5: Cache Statistics")
    print_info(f"Total requests: {mem.cache.stats['total_requests']}")
    print_info(f"Hits: {mem.cache.stats['hits']}")
    print_info(f"Misses: {mem.cache.stats['misses']}")
    print_info(f"Cache size: {mem.cache.stats['cache_size']}")
    
    if mem.cache.stats['total_requests'] > 0:
        hit_rate = mem.cache.stats['hits'] / mem.cache.stats['total_requests']
        print_info(f"Hit rate: {hit_rate:.2%}")
    
    print_section("6.6: Multiple Texts Caching")
    test_texts = [
        "Python is a great programming language",
        "Machine learning uses neural networks",
        "Space exploration is fascinating",
        "Cooking requires good ingredients",
    ]
    
    # First pass - all misses
    initial_misses = mem.cache.stats["misses"]
    for text in test_texts:
        mem._get_embedding(text)
    misses_after_first = mem.cache.stats["misses"]
    
    # Second pass - all hits
    initial_hits = mem.cache.stats["hits"]
    for text in test_texts:
        mem._get_embedding(text)
    hits_after_second = mem.cache.stats["hits"]
    
    assert_test(misses_after_first - initial_misses == len(test_texts),
                "All texts missed on first pass")
    assert_test(hits_after_second - initial_hits == len(test_texts),
                "All texts hit on second pass")
    print_success(f"Cached {len(test_texts)} embeddings successfully")


# ===== TEST 7: PERSISTENCE OPERATIONS =====

def test_persistence_operations():
    """Test save, load, merge, and snapshot functionality."""
    print_banner("Test 7: Persistence Operations")
    
    # Create memory with data
    mem = Memory(embedding_method="openai", enable_cache=True)
    
    print_section("7.1: Populate Memory")
    for conv_name, conv_data in REAL_CONVERSATIONS.items():
        for turn in conv_data:
            mem.insert(**turn)
    
    initial_count = len(mem.memories)
    print_success(f"Populated with {initial_count} memories")
    
    print_section("7.2: Save to Pickle")
    save_dir = Path(__file__).parent / "test_persistence"
    save_dir.mkdir(exist_ok=True)
    
    save_path = mem.save("test_save_pickle", directory=str(save_dir), format="pickle")
    assert_test(Path(f"{save_path}.pkl").exists(), "Pickle file created")
    assert_test(Path(f"{save_path}.json").exists(), "JSON file also created")
    assert_test(Path(f"{save_path}_metadata.json").exists(), "Metadata file created")
    print_success("Saved to pickle format")
    
    print_section("7.3: Load from Pickle")
    mem2 = Memory(embedding_method="openai", enable_cache=True)
    loaded_count = mem2.load("test_save_pickle", directory=str(save_dir))
    
    assert_test(loaded_count == initial_count, "Loaded correct number of memories")
    assert_test(len(mem2.memories) == initial_count, "Memory count matches")
    assert_test(mem2.embedding_matrix.shape[0] == initial_count, 
                "Embeddings loaded correctly")
    print_success(f"Loaded {loaded_count} memories from pickle")
    
    print_section("7.4: Verify Loaded Data")
    # Check that specific memory exists
    python_memories = mem2.get(sample_id="conv-python-learning")
    assert_test(len(python_memories) > 0, "Loaded memories contain expected data")
    
    # Verify embeddings work after loading
    results = mem2.search("Python programming", search_method="text-embedding", top_k=3)
    assert_test(len(results) > 0, "Search works after loading")
    print_success("Loaded data verified")
    
    print_section("7.5: Save to JSON")
    save_path_json = mem.save("test_save_json", directory=str(save_dir), format="json")
    assert_test(Path(f"{save_path_json}.json").exists(), "JSON file created")
    print_success("Saved to JSON format")
    
    print_section("7.6: Load from JSON")
    mem3 = Memory(embedding_method="openai", enable_cache=True)
    loaded_count = mem3.load("test_save_json", directory=str(save_dir), format="json")
    
    assert_test(loaded_count == initial_count, "Loaded correct number from JSON")
    # Note: JSON format doesn't save embeddings, so they need to be rebuilt from cache
    assert_test(mem3.embedding_matrix.shape[0] == initial_count,
                "Embeddings rebuilt from cache")
    print_success(f"Loaded {loaded_count} memories from JSON")
    
    print_section("7.7: Merge (Load without Clearing)")
    mem4 = Memory(embedding_method="openai", enable_cache=True)
    # Add some unique memories
    mem4.insert("conv-unique", 1, "1:00 PM", "User", "Unique content A")
    mem4.insert("conv-unique", 2, "1:01 PM", "AI", "Unique content B")
    initial_mem4_count = len(mem4.memories)
    
    # Load existing memories without clearing
    loaded_count = mem4.load("test_save_pickle", directory=str(save_dir), 
                             clear_existing=False)
    
    final_count = len(mem4.memories)
    assert_test(final_count > initial_mem4_count, "Memories merged")
    assert_test(final_count == initial_mem4_count + loaded_count,
                "Merge added all new memories")
    print_success(f"Merged: {initial_mem4_count} + {loaded_count} = {final_count}")
    
    print_section("7.8: List Saved Files")
    saves = mem.list_saves(directory=str(save_dir))
    assert_test(len(saves) >= 2, "Found saved files")
    print_success(f"Found {len(saves)} saved memory files")
    
    for save in saves:
        print_info(f"  - {save['save_name']}: {save['total_memories']} memories "
                  f"({save.get('format', 'unknown')} format)")
    
    print_section("7.9: Snapshot Testing")
    # Create a snapshot directory
    snapshot_dir = save_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    
    # Save epoch 0
    mem.save("epoch_0", directory=str(snapshot_dir))
    
    # Modify memory
    mem.insert("conv-new", 1, "5:00 PM", "User", "New memory after epoch 0")
    
    # Save epoch 1
    mem.save("epoch_1", directory=str(snapshot_dir))
    
    # Load epoch 0 to verify snapshot
    mem_epoch0 = Memory(embedding_method="openai", enable_cache=True)
    mem_epoch0.load("epoch_0", directory=str(snapshot_dir))
    
    # Load epoch 1
    mem_epoch1 = Memory(embedding_method="openai", enable_cache=True)
    mem_epoch1.load("epoch_1", directory=str(snapshot_dir))
    
    assert_test(len(mem_epoch1.memories) > len(mem_epoch0.memories),
                "Epoch 1 has more memories than epoch 0")
    print_success("Snapshot system working correctly")


# ===== TEST 8: MEMORY MANAGER INTEGRATION =====

def test_memory_manager_integration():
    """Test MemoryManager with JSON commands."""
    print_banner("Test 8: Memory Manager Integration")
    
    mem = Memory(embedding_method="openai", enable_cache=True)
    manager = MemoryManager(embedding_method="openai", enable_cache=True)
    
    print_section("8.1: Insert via JSON Command (Dict)")
    command = {
        "operation": "insert",
        "sample_id": "conv-manager-test",
        "session_id": 1,
        "session_time": "3:00 PM on December 29, 2024",
        "speaker": "TestUser",
        "content": "Testing memory manager insertion functionality"
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "success", "Insert command successful")
    assert_test("memory_id" in result, "Memory ID returned")
    memory_id = result["memory_id"]
    print_success(f"Inserted via manager: {memory_id}")
    
    print_section("8.2: Insert via JSON String")
    command_json = json.dumps({
        "operation": "insert",
        "sample_id": "conv-manager-test",
        "session_id": 2,
        "session_time": "3:02 PM on December 29, 2024",
        "speaker": "TestAI",
        "content": "Testing JSON string command parsing"
    })
    result = manager.execute_command(mem, command_json)
    assert_test(result["status"] == "success", "JSON string command successful")
    print_success("JSON string parsed correctly")
    
    print_section("8.3: Search via Manager")
    command = {
        "operation": "search",
        "query": "testing functionality",
        "sample_id": "conv-manager-test",
        "search_method": "text-embedding",
        "top_k": 5
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "success", "Search command successful")
    assert_test(len(result["data"]) > 0, "Search returned results")
    print_success(f"Found {len(result['data'])} results via manager")
    
    print_section("8.4: Update via Manager")
    command = {
        "operation": "update",
        "memory_id": memory_id,
        "content": "Updated content via memory manager"
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "success", "Update command successful")
    assert_test(result["data"]["content"] == "Updated content via memory manager",
                "Content updated correctly")
    print_success("Update via manager successful")
    
    print_section("8.5: Delete via Manager")
    command = {
        "operation": "delete",
        "memory_id": memory_id
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "success", "Delete command successful")
    
    # Verify deletion
    remaining = mem.get(sample_id="conv-manager-test")
    assert_test(all(m["memory_id"] != memory_id for m in remaining),
                "Memory deleted successfully")
    print_success("Delete via manager successful")
    
    print_section("8.6: Batch Command Execution")
    commands = [
        {
            "operation": "insert",
            "sample_id": "conv-batch",
            "session_id": 1,
            "session_time": "4:00 PM",
            "speaker": "User1",
            "content": "Batch insert 1"
        },
        {
            "operation": "insert",
            "sample_id": "conv-batch",
            "session_id": 2,
            "session_time": "4:01 PM",
            "speaker": "User2",
            "content": "Batch insert 2"
        },
        {
            "operation": "search",
            "query": "batch",
            "sample_id": "conv-batch",
            "search_method": "bm25"
        }
    ]
    
    result = manager.execute_batch(mem, commands)
    assert_test(result["status"] == "success", "Batch execution successful")
    assert_test(result["successful"] == 3, "All commands succeeded")
    print_success(f"Batch: {result['successful']}/{result['total_commands']} succeeded")
    
    print_section("8.7: Error Handling - Invalid Command")
    command = {
        "operation": "invalid_operation",
        "data": "some data"
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "error", "Invalid command returns error")
    print_success("Error handling working correctly")
    
    print_section("8.8: Error Handling - Missing Fields")
    command = {
        "operation": "insert",
        "sample_id": "conv-test"
        # Missing required fields
    }
    result = manager.execute_command(mem, command)
    assert_test(result["status"] == "error", "Missing fields returns error")
    print_success("Missing field validation working")


# ===== TEST 9: EDGE CASES AND STRESS TESTING =====

def test_edge_cases_and_stress():
    """Test edge cases and stress scenarios."""
    print_banner("Test 9: Edge Cases and Stress Testing")
    
    mem = Memory(embedding_method="openai", enable_cache=True)
    
    print_section("9.1: Empty Memory Operations")
    results = mem.search("test query", search_method="text-embedding")
    assert_test(len(results) == 0, "Search on empty memory returns empty")
    
    all_memories = mem.get()
    assert_test(len(all_memories) == 0, "Get on empty memory returns empty")
    
    success = mem.delete("nonexistent-id")
    assert_test(not success, "Delete on empty memory returns False")
    print_success("Empty memory operations handled correctly")
    
    print_section("9.2: Very Long Content")
    long_content = "This is a very long content. " * 200  # ~5000 chars
    result = mem.insert(
        sample_id="conv-long",
        session_id=1,
        session_time="5:00 PM",
        speaker="User",
        content=long_content
    )
    assert_test(result is not None, "Long content inserted successfully")
    assert_test(len(result["content"]) == len(long_content), "Content length preserved")
    print_success(f"Inserted {len(long_content)} character content")
    
    print_section("9.3: Special Characters")
    special_content = "Test with émojis 🎉🚀, spëcial çhars, and symbols: @#$%^&*()"
    result = mem.insert(
        sample_id="conv-special",
        session_id=1,
        session_time="5:01 PM",
        speaker="User",
        content=special_content
    )
    assert_test(result is not None, "Special characters handled")
    assert_test(result["content"] == special_content, "Special characters preserved")
    print_success("Special characters handled correctly")
    
    print_section("9.4: Stress Test - Many Insertions")
    start_time = time.time()
    stress_count = 50
    
    for i in range(stress_count):
        mem.insert(
            sample_id=f"conv-stress-{i % 10}",  # 10 conversations
            session_id=i + 1,
            session_time=f"5:{i:02d} PM",
            speaker=f"User{i % 5}",  # 5 different speakers
            content=f"Stress test message number {i} with some varied content about topic {i % 7}"
        )
    
    elapsed = time.time() - start_time
    assert_test(len(mem.memories) >= stress_count, 
                f"Inserted {stress_count} memories")
    print_success(f"Stress test: {stress_count} insertions in {elapsed:.2f}s "
                 f"({stress_count/elapsed:.1f} ops/sec)")
    
    print_section("9.5: Stress Test - Many Searches")
    start_time = time.time()
    search_count = 20
    
    queries = [
        "stress test message",
        "topic content",
        "varied information",
        "User message",
    ]
    
    for i in range(search_count):
        query = queries[i % len(queries)]
        results = mem.search(query, search_method="text-embedding", top_k=5)
    
    elapsed = time.time() - start_time
    print_success(f"Stress test: {search_count} searches in {elapsed:.2f}s "
                 f"({search_count/elapsed:.1f} searches/sec)")
    
    print_section("9.6: Concurrent Sample IDs")
    # Test that multiple conversations are isolated correctly
    mem.insert("conv-A", 1, "6:00 PM", "User", "Message in conversation A")
    mem.insert("conv-B", 1, "6:00 PM", "User", "Message in conversation B")
    
    conv_a = mem.get(sample_id="conv-A")
    conv_b = mem.get(sample_id="conv-B")
    
    assert_test(len(conv_a) > 0 and len(conv_b) > 0, "Both conversations exist")
    assert_test(all(m["sample_id"] == "conv-A" for m in conv_a),
                "Conversation A isolated")
    assert_test(all(m["sample_id"] == "conv-B" for m in conv_b),
                "Conversation B isolated")
    print_success("Conversation isolation working correctly")
    
    print_section("9.7: Embedding Dimension Consistency")
    # Verify all embeddings have the same dimension
    if mem.embedding_matrix.shape[0] > 0:
        dim = mem.embedding_matrix.shape[1]
        assert_test(dim == 1536, f"Correct embedding dimension: {dim}")
        
        # Check no NaN or infinite values
        assert_test(not np.isnan(mem.embedding_matrix).any(), 
                   "No NaN in embeddings")
        assert_test(not np.isinf(mem.embedding_matrix).any(), 
                   "No infinite values in embeddings")
        print_success("Embedding matrix validated")


# ===== TEST 10: COMPREHENSIVE INTEGRATION SCENARIO =====

def test_comprehensive_integration():
    """Full integration test simulating real usage."""
    print_banner("Test 10: Comprehensive Integration Scenario")
    
    print_section("10.1: Initialize System")
    cache_dir = Path(__file__).parent / "test_comprehensive_cache"
    mem = Memory(embedding_method="openai", enable_cache=True, cache_dir=str(cache_dir))
    manager = MemoryManager(embedding_method="openai", enable_cache=True)
    print_success("System initialized")
    
    print_section("10.2: Simulate Multi-turn Conversations")
    # Simulate 3 conversations with multiple turns
    for conv_name, conv_data in list(REAL_CONVERSATIONS.items())[:3]:
        print_info(f"Processing conversation: {conv_name}")
        for turn in conv_data:
            command = {
                "operation": "insert",
                **turn
            }
            result = manager.execute_command(mem, command)
            assert_test(result["status"] == "success", 
                       f"Turn inserted: {turn['speaker']}")
    
    total_memories = len(mem.memories)
    print_success(f"Inserted {total_memories} conversation turns")
    
    print_section("10.3: Perform Various Searches")
    search_queries = [
        ("Python programming basics", "text-embedding"),
        ("machine learning networks", "text-embedding"),
        ("pasta cooking tips", "bm25"),
    ]
    
    for query, method in search_queries:
        results = mem.search(query, search_method=method, top_k=3)
        print_info(f"Query '{query}' ({method}): {len(results)} results")
        if results:
            print_info(f"  Top result: {results[0][0]['content'][:60]}... "
                      f"(score: {results[0][1]:.4f})")
    
    print_section("10.4: Update Based on Search Results")
    # Find and update a memory
    results = mem.search("Python", search_method="text-embedding", top_k=1)
    if results:
        memory_to_update = results[0][0]
        memory_id = memory_to_update["memory_id"]
        
        update_command = {
            "operation": "update",
            "memory_id": memory_id,
            "content": "Updated: Python is an excellent language with rich ecosystem"
        }
        result = manager.execute_command(mem, update_command)
        assert_test(result["status"] == "success", "Update successful")
        print_success("Updated memory based on search results")
    
    print_section("10.5: Filter and Delete")
    # Delete all memories from a specific speaker in one conversation
    to_delete = mem.get(sample_id="conv-ml-discussion", speaker="Student")
    delete_count = len(to_delete)
    
    for memory in to_delete:
        command = {
            "operation": "delete",
            "memory_id": memory["memory_id"]
        }
        result = manager.execute_command(mem, command)
        assert_test(result["status"] == "success", "Delete successful")
    
    print_success(f"Deleted {delete_count} filtered memories")
    
    print_section("10.6: Save Snapshot")
    save_dir = Path(__file__).parent / "test_comprehensive_saves"
    save_dir.mkdir(exist_ok=True)
    
    save_path = mem.save("comprehensive_snapshot", directory=str(save_dir))
    print_success(f"Snapshot saved: {save_path}")
    
    print_section("10.7: Verify Cache Statistics")
    if mem.cache:
        stats = mem.cache.stats
        print_info(f"Cache hits: {stats['hits']}")
        print_info(f"Cache misses: {stats['misses']}")
        print_info(f"Cache size: {stats['cache_size']}")
        
        if stats['total_requests'] > 0:
            hit_rate = stats['hits'] / stats['total_requests']
            print_info(f"Hit rate: {hit_rate:.2%}")
            assert_test(hit_rate > 0, "Cache is being utilized")
    
    print_section("10.8: Load and Verify")
    mem2 = Memory(embedding_method="openai", enable_cache=True, cache_dir=str(cache_dir))
    loaded_count = mem2.load("comprehensive_snapshot", directory=str(save_dir))
    
    assert_test(loaded_count == len(mem.memories), "All memories loaded")
    
    # Verify search works after loading
    results = mem2.search("learning", search_method="text-embedding", top_k=5)
    assert_test(len(results) > 0, "Search works after loading")
    print_success("Snapshot loaded and verified")
    
    print_section("10.9: Final System State")
    print_info(f"Total memories: {len(mem.memories)}")
    print_info(f"Unique conversations: {len(set(m['sample_id'] for m in mem.memories))}")
    print_info(f"Unique speakers: {len(set(m['speaker'] for m in mem.memories))}")
    print_info(f"Embedding matrix shape: {mem.embedding_matrix.shape}")
    print_info(f"Cache enabled: {mem.enable_cache}")


# ===== MAIN TEST RUNNER =====

def run_all_tests():
    """Run all intensive integration tests."""
    print("\n" + "="*80)
    print("  INTENSIVE MEMORY SYSTEM INTEGRATION TEST SUITE")
    print("  Testing ALL functionalities with REAL examples")
    print("="*80)
    print(f"\nStarting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    failed_tests = []
    
    tests = [
        ("Basic Insertion", test_basic_insertion),
        ("Search Functionality", test_search_functionality),
        ("Retrieval Operations", test_retrieval_operations),
        ("Update Operations", test_update_operations),
        ("Delete Operations", test_delete_operations),
        ("Cache Functionality", test_cache_functionality),
        ("Persistence Operations", test_persistence_operations),
        ("Memory Manager Integration", test_memory_manager_integration),
        ("Edge Cases and Stress", test_edge_cases_and_stress),
        ("Comprehensive Integration", test_comprehensive_integration),
    ]
    
    # Run tests that return memory object first
    mem = None
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"Running Test {i}/{len(tests)}: {name}")
        print(f"{'='*80}")
        
        try:
            if i == 1:  # First test returns memory
                mem = test_func()
            elif i in [2, 3, 4, 5] and mem:  # These tests use the memory
                test_func(mem)
            else:  # Other tests are independent
                test_func()
            
            print(f"\n✓ {name} PASSED")
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed_tests.append((name, str(e)))
            import traceback
            traceback.print_exc()
    
    # Print summary
    elapsed = time.time() - start_time
    print_banner("TEST SUITE SUMMARY")
    
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {len(tests) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    else:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe memory system is working correctly across all functionalities:")
        print("  ✓ Insertion (single, bulk, duplicates)")
        print("  ✓ Search (text-embedding, BM25, filters)")
        print("  ✓ Retrieval (get with filters)")
        print("  ✓ Update (content, embeddings)")
        print("  ✓ Delete (single, bulk, cascade)")
        print("  ✓ Cache (hit/miss, persistence)")
        print("  ✓ Persistence (save, load, merge)")
        print("  ✓ Memory Manager (JSON commands)")
        print("  ✓ Edge cases and stress testing")
        print("  ✓ Comprehensive integration")
    
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_all_tests()
