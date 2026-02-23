"""
Unit tests for memory.py performance optimizations.

Tests cover:
1. Pre-allocated embedding matrix with growth factor (O(1) inserts)
2. Batched cosine similarity in _search_embedding
3. O(1) _embedding_id_to_idx dict for update/delete
4. BM25 removal
5. Save/load round-trip with pre-allocated matrix
6. Delete swap-with-last correctness
"""

import sys
import os
import json
import tempfile
import shutil
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', '..')
sys.path.insert(0, PROJECT_ROOT)


def _mock_embedding(dim=1536):
    """Create a deterministic mock embedding based on content hash."""
    def _get_embedding(self, text, method="openai"):
        np.random.seed(hash(text) % (2**31))
        emb = np.random.randn(dim).astype(np.float64)
        # Normalize so cosine similarity is meaningful
        return emb / np.linalg.norm(emb)
    return _get_embedding


class TestMemoryPreAllocatedMatrix(unittest.TestCase):
    """Test pre-allocated embedding matrix with growth factor."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_initial_capacity(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory, _INITIAL_EMBED_CAPACITY
        mem = Memory(enable_cache=False)
        
        # Check initial state
        self.assertEqual(mem._embed_capacity, _INITIAL_EMBED_CAPACITY)
        self.assertEqual(mem._embed_count, 0)
        self.assertEqual(mem.embedding_matrix.shape, (_INITIAL_EMBED_CAPACITY, 1536))

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_insert_updates_count_and_idx(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        result = mem.insert("conv-1", 1, "10am Dec 1", "Alice", "Hello there", "D1:1")
        mid = result["memory_id"]
        
        self.assertEqual(mem._embed_count, 1)
        self.assertIn(mid, mem._embedding_id_to_idx)
        self.assertEqual(mem._embedding_id_to_idx[mid], 0)
        self.assertEqual(len(mem.embedding_ids), 1)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_growth_on_capacity_exceeded(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory, _INITIAL_EMBED_CAPACITY
        mem = Memory(enable_cache=False)
        
        # Insert exactly _INITIAL_EMBED_CAPACITY + 1 items to trigger growth
        for i in range(_INITIAL_EMBED_CAPACITY + 1):
            mem.insert("conv-1", 1, "10am Dec 1", "Alice", f"Content item {i}", f"D1:{i}")
        
        self.assertEqual(mem._embed_count, _INITIAL_EMBED_CAPACITY + 1)
        self.assertEqual(mem._embed_capacity, _INITIAL_EMBED_CAPACITY * 2)
        self.assertEqual(mem.embedding_matrix.shape[0], _INITIAL_EMBED_CAPACITY * 2)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_multiple_inserts_consistency(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        ids = []
        for i in range(10):
            result = mem.insert("conv-1", 1, "10am Dec 1", "Alice", f"Content {i}", f"D1:{i}")
            ids.append(result["memory_id"])
        
        self.assertEqual(mem._embed_count, 10)
        self.assertEqual(len(mem._embedding_id_to_idx), 10)
        
        # Every ID should map to a unique index
        indices = set(mem._embedding_id_to_idx.values())
        self.assertEqual(len(indices), 10)
        
        # Indices should be 0..9
        self.assertEqual(indices, set(range(10)))


class TestBatchedCosimSearch(unittest.TestCase):
    """Test batched cosine similarity in _search_embedding."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_returns_results(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        mem.insert("conv-1", 1, "10am", "Alice", "The weather is sunny today", "D1:1")
        mem.insert("conv-1", 1, "10am", "Bob", "I like rainy weather", "D1:2")
        mem.insert("conv-1", 1, "10am", "Alice", "Programming is fun", "D1:3")
        
        results = mem.search("sunny weather", search_method="text-embedding")
        self.assertGreater(len(results), 0)
        
        # Results should be (turn_dict, score) tuples
        for turn, score in results:
            self.assertIsInstance(turn, dict)
            self.assertIsInstance(score, float)
            self.assertIn("memory_id", turn)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_sorted_descending(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        for i in range(5):
            mem.insert("conv-1", 1, "10am", "Alice", f"Different content {i}", f"D1:{i}")
        
        results = mem.search("Different content 3", search_method="text-embedding")
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_top_k(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        for i in range(10):
            mem.insert("conv-1", 1, "10am", "Alice", f"Content about topic {i}", f"D1:{i}")
        
        results = mem.search("topic", top_k=3, search_method="text-embedding")
        self.assertLessEqual(len(results), 3)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_min_score_filter(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        mem.insert("conv-1", 1, "10am", "Alice", "content A", "D1:1")
        
        # Very high min_score should filter everything
        results = mem.search("unrelated query xyz", min_score=0.99, search_method="text-embedding")
        # Could be 0 or more depending on random embeddings — just check it doesn't crash
        for _, score in results:
            self.assertGreaterEqual(score, 0.99)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_speaker_filter(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        mem.insert("conv-1", 1, "10am", "Alice", "Alice's memory A", "D1:1")
        mem.insert("conv-1", 1, "10am", "Bob", "Bob's memory B", "D1:2")
        
        results = mem.search("memory", speaker="Alice", search_method="text-embedding")
        for turn, _ in results:
            self.assertEqual(turn["speaker"], "Alice")

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_empty_memory(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        results = mem.search("query", search_method="text-embedding")
        self.assertEqual(len(results), 0)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_search_single_item(self):
        """Regression: ensure single-item search works (np.squeeze edge case)."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        mem.insert("conv-1", 1, "10am", "Alice", "Only one memory", "D1:1")
        # Use min_score=-1.0 to ensure result isn't filtered by negative cosine sim from mock
        results = mem.search("one memory", min_score=-1.0, search_method="text-embedding")
        self.assertEqual(len(results), 1)


class TestBM25Search(unittest.TestCase):
    """Test that BM25 search works (lazy — no pre-initialization)."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_bm25_search_works(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        mem.insert("conv-1", 1, "10am", "Alice", "the weather is sunny", "D1:1")
        mem.insert("conv-1", 1, "10am", "Alice", "programming python code", "D1:2")
        
        results = mem.search("sunny weather", search_method="bm25")
        self.assertGreater(len(results), 0)
        # BM25 results are (turn, score) tuples
        for turn, score in results:
            self.assertIsInstance(turn, dict)
            self.assertIsInstance(score, float)


class TestO1EmbeddingIdLookup(unittest.TestCase):
    """Test O(1) embedding_id_to_idx dict for update/delete."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_update_uses_dict_lookup(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        result = mem.insert("conv-1", 1, "10am", "Alice", "original content", "D1:1")
        mid = result["memory_id"]
        
        # Update should use O(1) lookup
        updated = mem.update(mid, "updated content", "D1:2")
        self.assertIsNotNone(updated)
        self.assertEqual(updated["content"], "updated content")
        
        # Embedding should be updated (different content -> different embedding)
        idx = mem._embedding_id_to_idx[mid]
        # Just verify the index is still valid
        self.assertLess(idx, mem._embed_count)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_delete_swap_with_last(self):
        """Test that delete correctly swaps with last element."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        r1 = mem.insert("conv-1", 1, "10am", "Alice", "First content", "D1:1")
        r2 = mem.insert("conv-1", 1, "10am", "Alice", "Second content", "D1:2")
        r3 = mem.insert("conv-1", 1, "10am", "Alice", "Third content", "D1:3")
        
        mid1, mid2, mid3 = r1["memory_id"], r2["memory_id"], r3["memory_id"]
        
        # Save embedding of mid3 (last) before delete
        idx3_before = mem._embedding_id_to_idx[mid3]
        embed3_before = mem.embedding_matrix[idx3_before].copy()
        
        # Delete mid1 (first) — should swap mid3 into slot 0
        self.assertTrue(mem.delete(mid1))
        
        # Verify state
        self.assertEqual(mem._embed_count, 2)
        self.assertNotIn(mid1, mem._embedding_id_to_idx)
        self.assertIn(mid2, mem._embedding_id_to_idx)
        self.assertIn(mid3, mem._embedding_id_to_idx)
        
        # mid3 should now be at index 0 (swapped into mid1's slot)
        self.assertEqual(mem._embedding_id_to_idx[mid3], 0)
        np.testing.assert_array_almost_equal(mem.embedding_matrix[0], embed3_before)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_delete_last_element(self):
        """Test deleting the last element (no swap needed)."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        r1 = mem.insert("conv-1", 1, "10am", "Alice", "First content", "D1:1")
        r2 = mem.insert("conv-1", 1, "10am", "Alice", "Second content", "D1:2")
        
        mid1, mid2 = r1["memory_id"], r2["memory_id"]
        
        # Delete last element
        self.assertTrue(mem.delete(mid2))
        self.assertEqual(mem._embed_count, 1)
        self.assertIn(mid1, mem._embedding_id_to_idx)
        self.assertNotIn(mid2, mem._embedding_id_to_idx)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_delete_then_search(self):
        """Ensure search works correctly after delete."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        r1 = mem.insert("conv-1", 1, "10am", "Alice", "Weather today is nice", "D1:1")
        r2 = mem.insert("conv-1", 1, "10am", "Alice", "Programming python code", "D1:2")
        r3 = mem.insert("conv-1", 1, "10am", "Alice", "Cooking pasta dinner", "D1:3")
        
        # Delete first
        mem.delete(r1["memory_id"])
        
        # Search should still work and only return remaining 2
        results = mem.search("pasta", min_score=-1.0, search_method="text-embedding")
        returned_ids = {t["memory_id"] for t, _ in results}
        self.assertNotIn(r1["memory_id"], returned_ids)
        self.assertEqual(len(returned_ids), 2)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_multiple_deletes(self):
        """Test multiple consecutive deletes maintain consistency."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        results = []
        for i in range(5):
            results.append(mem.insert("conv-1", 1, "10am", "Alice", f"Content {i}", f"D1:{i}"))
        
        # Delete from middle, then beginning, then end
        mem.delete(results[2]["memory_id"])  # middle
        mem.delete(results[0]["memory_id"])  # beginning
        mem.delete(results[4]["memory_id"])  # was end (now swapped)
        
        self.assertEqual(mem._embed_count, 2)
        self.assertEqual(len(mem._embedding_id_to_idx), 2)
        
        # Remaining should be results[1] and results[3]
        remaining_ids = set(mem._embedding_id_to_idx.keys())
        self.assertEqual(remaining_ids, {results[1]["memory_id"], results[3]["memory_id"]})


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Test that save/load works with the pre-allocated matrix."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_pickle_round_trip(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        for i in range(5):
            mem.insert("conv-1", 1, "10am", "Alice", f"Content {i}", f"D1:{i}")
        
        # Save original state
        original_count = mem._embed_count
        original_matrix = mem.embedding_matrix[:original_count].copy()
        original_ids = list(mem.embedding_ids)
        
        # Save to temp dir
        tmpdir = tempfile.mkdtemp()
        try:
            mem.save("test_save", directory=tmpdir, format="pickle")
            
            # Load into new Memory
            mem2 = Memory(enable_cache=False)
            mem2.load("test_save", directory=tmpdir, format="pickle")
            
            self.assertEqual(mem2._embed_count, original_count)
            self.assertEqual(mem2.embedding_ids, original_ids)
            np.testing.assert_array_almost_equal(
                mem2.embedding_matrix[:mem2._embed_count], original_matrix
            )
            
            # Verify _embedding_id_to_idx was rebuilt
            self.assertEqual(len(mem2._embedding_id_to_idx), original_count)
            for mid in original_ids:
                self.assertIn(mid, mem2._embedding_id_to_idx)
        finally:
            shutil.rmtree(tmpdir)

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_save_only_used_portion(self):
        """Verify that save() only writes the used portion, not the full pre-allocated matrix."""
        from verl.rema_trainer.memory.memory_core.memory import Memory
        import pickle
        
        mem = Memory(enable_cache=False)
        mem.insert("conv-1", 1, "10am", "Alice", "Content 1", "D1:1")
        mem.insert("conv-1", 1, "10am", "Alice", "Content 2", "D1:2")
        
        tmpdir = tempfile.mkdtemp()
        try:
            mem.save("test_save", directory=tmpdir, format="pickle")
            
            with open(os.path.join(tmpdir, "test_save.pkl"), "rb") as f:
                data = pickle.load(f)
            
            # Saved matrix should be (2, 1536), not (64, 1536)
            self.assertEqual(data['embedding_matrix'].shape[0], 2)
        finally:
            shutil.rmtree(tmpdir)


class TestGrowEmbeddingMatrix(unittest.TestCase):
    """Test the _grow_embedding_matrix method."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_grow_preserves_data(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        # Insert a few items
        for i in range(3):
            mem.insert("conv-1", 1, "10am", "Alice", f"Content {i}", f"D1:{i}")
        
        original_data = mem.embedding_matrix[:3].copy()
        original_capacity = mem._embed_capacity
        
        mem._grow_embedding_matrix()
        
        self.assertEqual(mem._embed_capacity, original_capacity * 2)
        np.testing.assert_array_almost_equal(mem.embedding_matrix[:3], original_data)


class TestDuplicateInsert(unittest.TestCase):
    """Ensure duplicate detection still works with new insert logic."""

    @patch.object(
        __import__('verl.rema_trainer.memory.memory_core.memory', fromlist=['Memory']).Memory,
        '_get_embedding', _mock_embedding()
    )
    def test_duplicate_not_added(self):
        from verl.rema_trainer.memory.memory_core.memory import Memory
        mem = Memory(enable_cache=False)
        
        r1 = mem.insert("conv-1", 1, "10am", "Alice", "Same content", "D1:1")
        r2 = mem.insert("conv-1", 1, "10am", "Alice", "Same content", "D1:2")
        
        # Should return existing, not create new
        self.assertEqual(r1["memory_id"], r2["memory_id"])
        self.assertEqual(mem._embed_count, 1)
        self.assertEqual(len(mem.memories), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
