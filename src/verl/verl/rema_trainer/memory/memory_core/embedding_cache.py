"""
Embedding cache system for OpenAI and other embedding providers.
Saves costs by caching embeddings on disk and avoiding duplicate API calls.
"""

from __future__ import annotations
import os
import json
import hashlib
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import time
from collections import OrderedDict


class EmbeddingCache:
    """
    Disk-based cache for text embeddings.
    
    Features:
    - Saves embeddings to disk to persist between sessions
    - Uses content hash as key to detect identical text
    - Supports multiple embedding methods (openai, sentence-transformers, etc.)
    - Tracks cache statistics (hits, misses, cost savings)
    """
    
    def __init__(self, cache_dir: str = None, enabled: bool = True):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files. 
                      Default priority:
                      1. Provided cache_dir parameter
                      2. EMBEDDING_CACHE_DIR environment variable
                      3. Current directory + '/cache'
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        
        if cache_dir is None:
            # Check for EMBEDDING_CACHE_DIR environment variable
            cache_dir = os.environ.get('EMBEDDING_CACHE_DIR')
            
            if cache_dir is None:
                # Use current directory + '/cache'
                cache_dir = os.path.join(os.getcwd(), 'cache')
        
        self.cache_dir = Path(cache_dir)
        
        if self.enabled:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # In-memory index for statistics only (NOT persisted to avoid race conditions)
            # Removed self.index as it creates a memory leak (we never clear it automatically) 
            # and we only need basic hit/miss counters.
            
            # In-memory fast cache to avoid repeated disk reads (LRU cache)
            self.memory_cache = OrderedDict()
            self.max_memory_cache_size = 50000
            
            # Statistics
            self.stats = {
                "hits": 0,
                "misses": 0,
                "total_requests": 0,
                "disk_hits": 0,    # New stat
                "ram_hits": 0      # New stat
            }
    
    # Files on disk are the source of truth, memory_cache is in-memory LRU
    
    def _compute_hash(self, text: str, method: str, model: str = None) -> str:
        """
        Compute a hash key for the text and embedding parameters.
        
        Args:
            text: The text to embed
            method: Embedding method (e.g., 'openai', 'sentence-transformers')
            model: Model name/version (optional)
        
        Returns:
            SHA256 hash as hex string
        """
        # Create a unique key based on text, method, and model
        key_parts = [text, method]
        if model:
            key_parts.append(model)
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_hash: str) -> Path:
        """Get the file path for a cached embedding."""
        # Use first 2 chars of hash for subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / cache_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{cache_hash}.npy"
    
    def get(self, text: str, method: str, model: str = None) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from cache if it exists.
        
        Args:
            text: The text to look up
            method: Embedding method
            model: Model name (optional)
        
        Returns:
            Cached embedding as numpy array, or None if not found
        """
        if not self.enabled:
            return None
        
        self.stats["total_requests"] += 1
        
        cache_hash = self._compute_hash(text, method, model)
        
        # Check in-memory fast cache first
        if hasattr(self, 'memory_cache') and cache_hash in self.memory_cache:
            self.stats["hits"] += 1
            self.stats["ram_hits"] += 1
            # Move to end to mark as recently used
            self.memory_cache.move_to_end(cache_hash)
            return self.memory_cache[cache_hash]
            
        cache_path = self._get_cache_path(cache_hash)
        
        # Check disk directly using try/except to avoid an extra network filesystem stat() call
        try:
            embedding = np.load(cache_path)
            self.stats["hits"] += 1
            self.stats["disk_hits"] += 1
            
            # Add to in-memory cache
            if hasattr(self, 'memory_cache'):
                self.memory_cache[cache_hash] = embedding
                if len(self.memory_cache) > self.max_memory_cache_size:
                    self.memory_cache.popitem(last=False)
            
            return embedding
        except OSError:
            # FileNotFoundError or other read errors (cache miss is expected during new rollouts)
            self.stats["misses"] += 1
            return None
        except (EOFError, ValueError):
            # Corrupted cache file (e.g. empty file from interrupted write) — delete and treat as miss
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.stats["misses"] += 1
            return None
    
    def set(self, text: str, embedding: np.ndarray, method: str, model: str = None, cost: float = 0.0):
        """
        Store an embedding in the cache.
        
        Args:
            text: The text that was embedded
            embedding: The embedding vector as numpy array
            method: Embedding method used
            model: Model name (optional)
            cost: API cost for this embedding (for statistics)
        """
        if not self.enabled:
            return
        
        cache_hash = self._compute_hash(text, method, model)
        cache_path = self._get_cache_path(cache_hash)
        
        try:
            # Save embedding to disk
            np.save(cache_path, embedding)
            
            # Update in-memory fast cache
            if hasattr(self, 'memory_cache'):
                self.memory_cache[cache_hash] = embedding
                if len(self.memory_cache) > self.max_memory_cache_size:
                    self.memory_cache.popitem(last=False)
            
        except Exception as e:
            print(f"Warning: Failed to cache embedding: {e}")
    
    def clear(self, method: str = None):
        """
        Clear the cache (Disk clearing omitted for performance and safety during multi-worker runs).
        Only RAM cache is cleared here.
        """
        if not self.enabled:
            return
            
        if hasattr(self, 'memory_cache'):
            self.memory_cache.clear()
            print("RAM cache cleared.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics including hit rate and cost savings
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        if stats["total_requests"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["total_requests"]
        else:
            stats["hit_rate"] = 0.0
        
        # Calculate estimated cost savings no longer accurate without self.index
        stats["estimated_cost_saved"] = 0.0
        stats["enabled"] = self.enabled
        
        # Add cache sizes
        if hasattr(self, 'memory_cache'):
            stats["ram_cache_size"] = len(self.memory_cache)
        
        return stats
    
    def get_cache_info(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get detailed cache information.
        
        Args:
            top_n: Number of most recent entries to show
        
        Returns:
            Dict with cache details
        """
        return {
            "total_entries_in_ram": len(self.memory_cache) if hasattr(self, 'memory_cache') else 0,
            "cache_dir": str(self.cache_dir),
            "stats": self.get_stats()
        }
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """
        Cleanup disabled to avoid iterating over cluster disk contents.
        """
        pass


# Global cache instance
_global_cache = None


def get_cache(cache_dir: str = None, enabled: bool = True) -> EmbeddingCache:
    """
    Get or create the global embedding cache instance.
    
    Args:
        cache_dir: Directory for cache (only used on first call)
        enabled: Whether to enable caching
    
    Returns:
        EmbeddingCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = EmbeddingCache(cache_dir=cache_dir, enabled=enabled)
    
    return _global_cache
