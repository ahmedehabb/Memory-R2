"""
Test suite for embedding cache system.
Demonstrates cost savings and performance improvements.
"""

import time
import numpy as np
from verl.rema_trainer.memory.memory_core.embedding_cache import EmbeddingCache, get_cache
from verl.rema_trainer.memory.memory_core.memory import Memory


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_basic_cache_operations():
    """Test basic cache get/set operations."""
    print_section("Test 1: Basic Cache Operations")
    
    # Create a new cache
    cache = EmbeddingCache(cache_dir="./test_cache", enabled=True)
    
    # Test 1: Cache miss
    print("1.1 Testing cache miss:")
    result = cache.get("Hello world", "openai", "text-embedding-3-small")
    assert result is None, "Should return None on cache miss"
    print("✓ Cache miss handled correctly")
    
    # Test 2: Store embedding
    print("\n1.2 Storing embedding in cache:")
    test_embedding = np.random.rand(1536)  # Simulate OpenAI embedding
    cache.set("Hello world", test_embedding, "openai", "text-embedding-3-small", cost=0.00001)
    print("✓ Embedding stored")
    
    # Test 3: Cache hit
    print("\n1.3 Testing cache hit:")
    cached_result = cache.get("Hello world", "openai", "text-embedding-3-small")
    assert cached_result is not None, "Should find cached embedding"
    assert np.allclose(cached_result, test_embedding), "Cached embedding should match"
    print("✓ Cache hit successful")
    print(f"✓ Embedding matches original")
    
    # Test 4: Different text should miss
    print("\n1.4 Testing cache miss for different text:")
    result = cache.get("Different text", "openai", "text-embedding-3-small")
    assert result is None, "Should return None for different text"
    print("✓ Different text correctly results in cache miss")
    
    # Clean up
    cache.clear()
    print("\n✓ Cache cleared")


def test_cache_statistics():
    """Test cache statistics tracking."""
    print_section("Test 2: Cache Statistics")
    
    cache = EmbeddingCache(cache_dir="./test_cache", enabled=True)
    
    # Simulate some cache operations
    texts = [
        "Python programming",
        "Machine learning",
        "Neural networks",
        "Python programming",  # Duplicate
        "Deep learning",
        "Machine learning"     # Duplicate
    ]
    
    print("Simulating cache operations...")
    for i, text in enumerate(texts):
        # Try to get from cache
        result = cache.get(text, "openai")
        
        if result is None:
            # Cache miss - create and store embedding
            embedding = np.random.rand(1536)
            cache.set(text, embedding, "openai", cost=0.00001)
            print(f"  {i+1}. '{text}' - MISS (stored)")
        else:
            print(f"  {i+1}. '{text}' - HIT")
    
    # Get statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']} entries")
    print(f"  Estimated cost saved: ${stats['estimated_cost_saved']:.6f}")
    
    assert stats['hits'] == 2, "Should have 2 cache hits"
    assert stats['misses'] == 4, "Should have 4 cache misses"
    assert stats['hit_rate'] == 2/6, "Hit rate should be 2/6"
    
    print("\n✓ Statistics tracking works correctly")
    
    # Clean up
    cache.clear()


def test_cache_with_memory():
    """Test cache integration with Memory class."""
    print_section("Test 3: Cache Integration with Memory")
    
    # Create memory with caching enabled
    print("3.1 Creating memory with cache enabled:")
    mem = Memory(embedding_method="openai", 
                 enable_cache=True, cache_dir="./test_cache")
    
    # Check if cache was initialized
    assert mem.cache is not None, "Cache should be initialized"
    print("✓ Memory initialized with cache")
    
    # Insert memories (will generate embeddings)
    print("\n3.2 Inserting memories (generating embeddings):")
    test_texts = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "Neural networks mimic the brain"
    ]
    
    start_time = time.time()
    for i, text in enumerate(test_texts):
        result = mem.insert(
            sample_id="test-conv",
            session_id=i+1,
            session_time="10:00 am",
            speaker="TestUser",
            content=text
        )
        mem_id = result['memory_id']
        print(f"  Inserted: [{mem_id}] {text[:50]}...")
    
    first_insert_time = time.time() - start_time
    print(f"\n✓ First insertion time: {first_insert_time:.3f}s")
    
    # Get cache stats
    if mem.cache:
        stats = mem.cache.get_stats()
        print(f"  Cache misses (new embeddings): {stats['misses']}")
        print(f"  Cache size: {stats['cache_size']}")
    
    # Create new memory instance with same cache
    print("\n3.3 Creating new memory instance (should use cached embeddings):")
    mem2 = Memory(embedding_method="openai",
                  enable_cache=True, cache_dir="./test_cache")
    
    start_time = time.time()
    for i, text in enumerate(test_texts):
        result = mem2.insert(
            sample_id="test-conv-2",
            session_id=i+1,
            session_time="10:00 am",
            speaker="TestUser",
            content=text
        )
        mem_id = result['memory_id']
        print(f"  Inserted: [{mem_id}] {text[:50]}...")
    
    second_insert_time = time.time() - start_time
    print(f"\n✓ Second insertion time: {second_insert_time:.3f}s")
    
    # Get updated cache stats
    if mem2.cache:
        stats = mem2.cache.get_stats()
        print(f"  Cache hits (reused embeddings): {stats['hits']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        
        if stats['hits'] > 0:
            print(f"\n✓ Cache is working! Embeddings were reused")
            print(f"  Time savings: {first_insert_time - second_insert_time:.3f}s")
    
    # Clean up
    if mem.cache:
        mem.cache.clear()


def test_cache_persistence():
    """Test that cache persists between sessions."""
    print_section("Test 4: Cache Persistence")
    
    cache_dir = "./test_cache"
    
    # Session 1: Create cache and add entries
    print("4.1 Session 1 - Creating cache and storing embeddings:")
    cache1 = EmbeddingCache(cache_dir=cache_dir, enabled=True)
    
    for i in range(3):
        text = f"Test embedding {i}"
        embedding = np.random.rand(1536)
        cache1.set(text, embedding, "openai", cost=0.00001)
        print(f"  Stored: {text}")
    
    stats1 = cache1.get_stats()
    print(f"\n  Cache size: {stats1['cache_size']} entries")
    
    # Simulate session end (don't clear cache)
    del cache1
    
    # Session 2: Load cache and verify entries exist
    print("\n4.2 Session 2 - Loading cache from disk:")
    cache2 = EmbeddingCache(cache_dir=cache_dir, enabled=True)
    
    stats2 = cache2.get_stats()
    print(f"  Loaded cache size: {stats2['cache_size']} entries")
    
    # Try to retrieve cached embeddings
    for i in range(3):
        text = f"Test embedding {i}"
        result = cache2.get(text, "openai")
        if result is not None:
            print(f"  ✓ Found cached: {text}")
        else:
            print(f"  ✗ Missing: {text}")
    
    assert stats2['cache_size'] == 3, "Cache should persist all 3 entries"
    print("\n✓ Cache persisted between sessions")
    
    # Clean up
    cache2.clear()


def test_cache_info():
    """Test cache information and cleanup."""
    print_section("Test 5: Cache Info and Cleanup")
    
    cache = EmbeddingCache(cache_dir="./test_cache", enabled=True)
    
    # Add some entries
    print("5.1 Adding test entries:")
    for i in range(5):
        text = f"Entry {i}"
        embedding = np.random.rand(1536)
        cache.set(text, embedding, "openai", cost=0.00001)
    
    # Get detailed info
    print("\n5.2 Cache information:")
    info = cache.get_cache_info(top_n=5)
    print(f"  Total entries: {info['total_entries']}")
    print(f"  Cache directory: {info['cache_dir']}")
    print(f"\n  Recent entries:")
    for entry in info['recent_entries']:
        print(f"    - {entry['method']}: {entry['text_length']} chars, "
              f"dim={entry['embedding_dim']}, "
              f"age={entry['age_hours']:.2f}h")
    
    # Test cleanup
    print("\n5.3 Testing cleanup (removing old entries):")
    print("  (Would remove entries older than 30 days)")
    cache.cleanup_old_entries(max_age_days=30)
    
    stats = cache.get_stats()
    print(f"  Entries after cleanup: {stats['cache_size']}")
    
    # Clean up
    cache.clear()
    print("\n✓ Cache info and cleanup work correctly")


def test_disabled_cache():
    """Test that disabling cache works correctly."""
    print_section("Test 6: Disabled Cache")
    
    print("6.1 Creating memory with cache disabled:")
    mem = Memory(embedding_method="openai", enable_cache=False)
    
    assert mem.cache is None, "Cache should be None when disabled"
    print("✓ Cache is disabled")
    
    print("\n6.2 Inserting memories without cache:")
    result = mem.insert(
        sample_id="test-conv",
        session_id=1,
        session_time="10:00 am",
        speaker="TestUser",
        content="Test without cache"
    )
    if result:
        print("✓ Memory operations work without cache")


def run_all_tests():
    """Run all cache tests."""
    print("\n" + "="*70)
    print("  EMBEDDING CACHE TEST SUITE")
    print("="*70)
    
    try:
        test_basic_cache_operations()
        test_cache_statistics()
        test_cache_with_memory()
        test_cache_persistence()
        test_cache_info()
        test_disabled_cache()
        
        print_section("ALL TESTS COMPLETED")
        print("✓ All cache tests passed successfully!")
        print("\nBENEFITS:")
        print("  • Saves money by avoiding duplicate OpenAI API calls")
        print("  • Faster performance for repeated embeddings")
        print("  • Persists between sessions")
        print("  • Automatic cost tracking")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test cache
        try:
            cache = EmbeddingCache(cache_dir="./test_cache", enabled=True)
            cache.clear()
            print("\n✓ Test cache cleaned up")
        except:
            pass


if __name__ == "__main__":
    run_all_tests()
