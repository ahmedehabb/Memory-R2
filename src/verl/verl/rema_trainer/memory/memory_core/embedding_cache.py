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
            self.index = {}
            
            # Statistics
            self.stats = {
                "hits": 0,
                "misses": 0,
                "total_requests": 0,
                "cache_size": 0
            }
    
    # Index methods removed - no disk persistence to avoid race conditions in parallel processes
    # Files on disk are the source of truth, index is in-memory only for statistics
    
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
        cache_path = self._get_cache_path(cache_hash)
        
        # Check disk directly - files are source of truth, not index!
        if not cache_path.exists():
            self.stats["misses"] += 1
            print(f"Warning: Cache miss for hash: {cache_hash}")
            return None
        
        try:
            embedding = np.load(cache_path)
            self.stats["hits"] += 1
            
            # Update in-memory index for statistics only (not persisted)
            self.index[cache_hash] = {
                "method": method,
                "model": model,
                "text_length": len(text),
                "embedding_dim": len(embedding) if embedding.ndim == 1 else embedding.shape[-1],
                "last_accessed": time.time(),
                "cache_file": str(cache_path)
            }
            self.stats["cache_size"] = len(self.index)
            
            return embedding
        except Exception as e:
            print(f"Warning: Failed to load cached embedding: {e}")
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
            
            # Update in-memory index for statistics only (not persisted)
            self.index[cache_hash] = {
                "method": method,
                "model": model,
                "text_length": len(text),
                "embedding_dim": len(embedding) if embedding.ndim == 1 else embedding.shape[-1],
                "created": time.time(),
                "last_accessed": time.time(),
                "cost": cost,
                "cache_file": str(cache_path)
            }
            self.stats["cache_size"] = len(self.index)
            
        except Exception as e:
            print(f"Warning: Failed to cache embedding: {e}")
    
    def clear(self, method: str = None):
        """
        Clear the cache.
        
        Args:
            method: If specified, only clear embeddings from this method
        """
        if not self.enabled:
            return
        
        if method is None:
            # Clear all
            for cache_info in self.index.values():
                cache_file = Path(cache_info["cache_file"])
                if cache_file.exists():
                    cache_file.unlink()
            
            self.index = {}
        else:
            # Clear only specific method
            to_remove = []
            for cache_hash, cache_info in self.index.items():
                if cache_info["method"] == method:
                    cache_file = Path(cache_info["cache_file"])
                    if cache_file.exists():
                        cache_file.unlink()
                    to_remove.append(cache_hash)
            
            for cache_hash in to_remove:
                del self.index[cache_hash]
        
        self.stats["cache_size"] = len(self.index)
        print(f"Cache cleared (in-memory index). Remaining entries: {len(self.index)}")
    
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
        
        # Calculate estimated cost savings (for OpenAI embeddings)
        # Assuming text-embedding-3-small costs $0.00002 per 1K tokens
        # Average ~4 chars per token
        total_saved_cost = 0.0
        for cache_info in self.index.values():
            if cache_info["method"] == "openai":
                total_saved_cost += cache_info.get("cost", 0.0)
        
        stats["estimated_cost_saved"] = total_saved_cost
        stats["enabled"] = self.enabled
        
        return stats
    
    def get_cache_info(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get detailed cache information.
        
        Args:
            top_n: Number of most recent entries to show
        
        Returns:
            Dict with cache details
        """
        # Sort by last accessed time
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1]["last_accessed"],
            reverse=True
        )
        
        recent_entries = []
        for cache_hash, info in sorted_entries[:top_n]:
            recent_entries.append({
                "hash": cache_hash[:8] + "...",
                "method": info["method"],
                "model": info.get("model"),
                "text_length": info["text_length"],
                "embedding_dim": info["embedding_dim"],
                "age_hours": (time.time() - info["created"]) / 3600,
                "last_accessed_hours_ago": (time.time() - info["last_accessed"]) / 3600
            })
        
        return {
            "total_entries": len(self.index),
            "recent_entries": recent_entries,
            "cache_dir": str(self.cache_dir),
            "stats": self.get_stats()
        }
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """
        Remove cache entries older than specified days.
        
        Args:
            max_age_days: Maximum age in days before removal
        """
        if not self.enabled:
            return
        
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        
        to_remove = []
        for cache_hash, cache_info in self.index.items():
            age = current_time - cache_info["last_accessed"]
            if age > max_age_seconds:
                cache_file = Path(cache_info["cache_file"])
                if cache_file.exists():
                    cache_file.unlink()
                to_remove.append(cache_hash)
        
        for cache_hash in to_remove:
            del self.index[cache_hash]
        
        self.stats["cache_size"] = len(self.index)
        
        print(f"Cleaned up {len(to_remove)} old entries from disk and in-memory index")


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
