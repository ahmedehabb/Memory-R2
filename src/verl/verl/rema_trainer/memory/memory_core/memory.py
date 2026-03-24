from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import json
import os
import openai
import uuid
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from verl.rema_trainer.memory.memory_core.embedding_cache import get_cache

# Load env once at module level instead of per-embedding call
load_dotenv()

_INITIAL_EMBED_CAPACITY = 64  # Pre-allocation size for embedding matrix

class Memory:
    """Stores LLM-generated memory items (summaries/facts extracted from conversations) in RAM."""

    EMBEDDING_METHOD = "openai"

    def __init__(self, embedding_method: str = None, enable_cache: bool = True, cache_dir: str = None) -> None:
        # Single memory store: LLM-generated memory items with metadata
        # Each turn: {
        #     "memory_id": str,  # unique identifier for this memory entry
        #     "sample_id": str,  # conversation ID (e.g., "conv-41")
        #     "session_id": int,
        #     "session_time": str,
        #     "speaker": str,
        #     "content": str
        # }
        self.memories: List[Dict[str, any]] = []
        
        # Set embedding method
        self.embedding_method = embedding_method or self.EMBEDDING_METHOD
        
        # Initialize embedding cache
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = get_cache(cache_dir=cache_dir, enabled=True)
        else:
            self.cache = None

        if self.embedding_method == "openai":
            self._embedding_dim = 1536  # text-embedding-3-small
        else:
            raise NotImplementedError(f"Embedding method '{self.embedding_method}' is not implemented yet.")
        
        # Embeddings stored as pre-allocated matrix for O(1) inserts
        self._embed_capacity = _INITIAL_EMBED_CAPACITY
        self.embedding_matrix: np.ndarray = np.empty((self._embed_capacity, self._embedding_dim))
        self._embed_count: int = 0  # Number of actual embeddings stored
        # Memory ID mappings to track which row corresponds to which memory
        self.embedding_ids: List[str] = []  # memory_ids
        self._embedding_id_to_idx: Dict[str, int] = {}  # O(1) lookup: memory_id -> matrix row index
        
        # Track all dia_ids that have been inserted/updated for easy evaluation
        self.dia_ids_set: set = set()
        
        # Track total tokens saved in memory
        self.total_tokens: int = 0

    def _generate_memory_id(self) -> str:
        """Generate a unique ID for a memory item."""
        return str(uuid.uuid4())[:8]  # Using first 8 characters of UUID

    def _get_embedding(self, text: str, method: str = "openai") -> np.ndarray:
        """Generate embedding for text using specified method.
        
        Args:
            text: Text to embed
            method: Embedding method to use. Options:
                - 'openai': OpenAI API (requires API key and costs money)
        
        Returns:
            numpy array of embeddings
        """
        # Try to get from cache first
        if self.enable_cache and self.cache is not None:
            cached_embedding = self.cache.get(text, method, model="text-embedding-3-small")
            if cached_embedding is not None:
                return cached_embedding
        
        if method == "openai":
            try:
                client = openai.OpenAI()
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                if not hasattr(self, '_embedding_dim'):
                    self._embedding_dim = 1536  # OpenAI text-embedding-3-small dimension
                
                embedding = np.array(response.data[0].embedding)
                
                # Cache the embedding
                if self.enable_cache and self.cache is not None:
                    # Estimate cost: $0.00002 per 1K tokens, roughly 4 chars per token
                    estimated_tokens = len(text) / 4
                    cost = (estimated_tokens / 1000) * 0.00002
                    self.cache.set(text, embedding, method, model="text-embedding-3-small", cost=cost)
                
                return embedding
            except Exception as e:
                print(f"Error generating OpenAI embedding: {e}")
                print("Make sure OPENAI_API_KEY is set in .env file")
                return np.zeros(1536)
        else:
            raise ValueError(f"Unknown embedding method: {method}. Use 'openai'.")
    
    def _get_embeddings_batch(self, texts: List[str], method: str = "openai") -> List[np.ndarray]:
        """Generate embeddings for multiple texts in a single API call.
        
        Checks cache first for each text. Only texts with cache misses are sent
        to the API in one batch request, reducing network round-trips.
        
        Args:
            texts: List of texts to embed
            method: Embedding method to use
        
        Returns:
            List of numpy arrays (same order as input texts)
        """
        if not texts:
            return []
        
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if self.enable_cache and self.cache is not None:
                cached = self.cache.get(text, method, model="text-embedding-3-small")
                if cached is not None:
                    results[i] = cached
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)
        
        # Batch API call for cache misses
        if uncached_texts and method == "openai":
            try:
                client = openai.OpenAI()
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=uncached_texts
                )
                # Response data is in same order as input
                for j, data_item in enumerate(response.data):
                    embedding = np.array(data_item.embedding)
                    orig_idx = uncached_indices[j]
                    results[orig_idx] = embedding
                    
                    # Cache each embedding
                    if self.enable_cache and self.cache is not None:
                        text = uncached_texts[j]
                        estimated_tokens = len(text) / 4
                        cost = (estimated_tokens / 1000) * 0.00002
                        self.cache.set(text, embedding, method, model="text-embedding-3-small", cost=cost)
            except Exception as e:
                print(f"Error in batch embedding API call: {e}")
                # Fall back to zeros for failed embeddings
                for j in range(len(uncached_texts)):
                    if results[uncached_indices[j]] is None:
                        results[uncached_indices[j]] = np.zeros(self._embedding_dim)
        
        # Fill any remaining None entries
        for i in range(len(results)):
            if results[i] is None:
                results[i] = np.zeros(self._embedding_dim)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    # --------------------------------------------------
    # Conversation Memory Operations
    # --------------------------------------------------
    
    def insert(self, sample_id: str, session_id: int, session_time: str, 
               speaker: str, content: str, dia_id: str) -> Dict[str, any]:
        """
        Insert a memory item (LLM-generated summary) with metadata.
        
        Args:
            sample_id: Conversation ID (e.g., "conv-41")
            session_id: Session number within the conversation
            session_time: Timestamp of the session (e.g., "11:01 am on 17 December, 2022")
            speaker: Name of the speaker
            content: LLM-generated content to save (not raw text from data)
            dia_id: Dialogue ID (e.g., "D3:6") to track source
            
        Returns:
            Dict with the inserted turn data including memory_id
        """
        # Check for duplicates: same sample_id, speaker, and content
        for existing_memory in self.memories:
            if (existing_memory["sample_id"] == sample_id and
                existing_memory["speaker"] == speaker and
                existing_memory["content"] == content):
                # Keep evidence coverage and temporal freshness when content repeats.
                if "dia_ids" not in existing_memory:
                    existing_memory["dia_ids"] = []
                if dia_id not in existing_memory["dia_ids"]:
                    existing_memory["dia_ids"].append(dia_id)
                    self.dia_ids_set.add(dia_id)

                # Backfill temporal provenance for older snapshots.
                existing_memory.setdefault("first_session_id", existing_memory.get("session_id", session_id))
                existing_memory.setdefault("first_session_time", existing_memory.get("session_time", session_time))
                existing_memory.setdefault("last_session_id", existing_memory.get("session_id", session_id))
                existing_memory.setdefault("last_session_time", existing_memory.get("session_time", session_time))
                if "mention_history" not in existing_memory:
                    existing_memory["mention_history"] = []

                existing_session_id = existing_memory.get("last_session_id")
                if isinstance(existing_session_id, int) and session_id >= existing_session_id:
                    existing_memory["last_session_id"] = session_id
                    existing_memory["last_session_time"] = session_time

                mention_key = (dia_id, session_id, session_time)
                seen_mentions = {
                    (m.get("dia_id"), m.get("session_id"), m.get("session_time"))
                    for m in existing_memory["mention_history"]
                    if isinstance(m, dict)
                }
                if mention_key not in seen_mentions:
                    existing_memory["mention_history"].append({
                        "dia_id": dia_id,
                        "session_id": session_id,
                        "session_time": session_time
                    })
                    # Keep bounded provenance to avoid prompt/cache bloat.
                    if len(existing_memory["mention_history"]) > 20:
                        existing_memory["mention_history"] = existing_memory["mention_history"][-20:]

                # Return existing memory instead of creating a duplicate row.
                return existing_memory
        
        memory_id = self._generate_memory_id()
        
        turn_data = {
            "memory_id": memory_id,
            "sample_id": sample_id,
            # Keep original session fields for backward compatibility.
            "session_id": session_id,
            "session_time": session_time,
            # Explicit temporal provenance to avoid lossy overwrite semantics.
            "first_session_id": session_id,
            "first_session_time": session_time,
            "last_session_id": session_id,
            "last_session_time": session_time,
            "speaker": speaker,
            "content": content,
            "dia_ids": [dia_id],  # Store dia_id in array
            "mention_history": [{
                "dia_id": dia_id,
                "session_id": session_id,
                "session_time": session_time
            }]
        }
        
        # Generate and store embedding for the content FIRST (atomic update)
        # If this fails, we haven't modified state yet
        embedding = self._get_embedding(content, method=self.embedding_method)
        try:
            # Grow matrix if at capacity
            if self._embed_count >= self._embed_capacity:
                self._grow_embedding_matrix()
            self.embedding_matrix[self._embed_count] = embedding
            self.embedding_ids.append(memory_id)
            self._embedding_id_to_idx[memory_id] = self._embed_count
            self._embed_count += 1
        except Exception as e:
            # If matrix update fails, rollback/don't proceed
            raise RuntimeError(f"Failed to update embedding matrix for insert: {e}")

        # Now update python objects
        self.memories.append(turn_data)
        
        # Track dia_id in the set for easy evaluation
        self.dia_ids_set.add(dia_id)
        
        # Update total tokens count
        self.total_tokens += len(self._tokenize(content))
        
        return turn_data
    
    def get(self, sample_id: str = None, speaker: str = None) -> List[Dict[str, any]]:
        """
        Retrieve memory items, optionally filtered by sample_id and/or speaker.
        
        Args:
            sample_id: Optional conversation ID to filter by
            speaker: Optional speaker name to filter by
            
        Returns:
            List of conversation turn dicts
        """
        if sample_id is None and speaker is None:
            return self.memories
        
        results = []
        for turn in self.memories:
            # Check filters
            if sample_id is not None and turn["sample_id"] != sample_id:
                continue
            if speaker is not None and turn["speaker"] != speaker:
                continue
            results.append(turn)
        
        return results
    
    def search(self, query: str, sample_id: str = None, speaker: str = None,
               top_k: int = None, min_score: float = 0.0, 
               search_method: str = "text-embedding") -> List[Tuple[Dict[str, any], float]]:
        """
        Search memory items using semantic similarity or BM25.
        Can filter by sample_id and/or speaker.
        
        Args:
            query: Search query
            sample_id: Optional filter by conversation ID
            speaker: Optional filter by speaker name
            top_k: Maximum number of results
            min_score: Minimum similarity score
            search_method: "text-embedding" or "bm25"
            
        Returns:
            List of tuples (turn_dict, score) sorted by score descending
        """
        # Filter by sample_id and/or speaker if specified
        filtered_turns = self.get(sample_id=sample_id, speaker=speaker)
        
        # Defensive type check for query
        if not isinstance(query, str):
            print(f"ERROR: search() received non-string query. Type: {type(query)}, value: {query}")
            return []
        
        if not filtered_turns or not query.strip():
            return []
        
        if search_method == "text-embedding":
            return self._search_embedding(filtered_turns, query, top_k, min_score)
        elif search_method == "bm25":
            return self._search_bm25(filtered_turns, query, top_k, min_score)
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'bm25' or 'text-embedding'.")
    
    def _search_bm25(self, turns: List[Dict[str, any]], query: str,
                     top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, any], float]]:
        """Search memory items using BM25 (lazy — index built only when called)."""
        if not turns:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Tokenize all turn contents
        tokenized_corpus = [self._tokenize(turn["content"]) for turn in turns]
        
        # Create BM25 object (lazy, only when bm25 search is requested)
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)
        
        # Create results
        results = []
        for i, turn in enumerate(turns):
            score = doc_scores[i]
            if score >= min_score:
                results.append((turn, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def _search_embedding(self, turns: List[Dict[str, any]], query: str,
                          top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, any], float]]:
        """Search memory items using batched embedding similarity."""
        if not turns:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query, method=self.embedding_method)
        if np.allclose(query_embedding, 0):
            return []
        
        # Build a single matrix of filtered embeddings for batched cosine similarity
        valid_turns = []
        valid_indices = []
        for turn in turns:
            memory_id = turn["memory_id"]
            if memory_id in self._embedding_id_to_idx:
                valid_turns.append(turn)
                valid_indices.append(self._embedding_id_to_idx[memory_id])
        
        if not valid_turns:
            return []
        
        # Single batched cosine similarity computation
        filtered_matrix = self.embedding_matrix[valid_indices]  # (N, dim)
        query_vec = query_embedding.reshape(1, -1)  # (1, dim)
        
        # Manual cosine similarity (avoids sklearn overhead)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        filtered_norms = np.linalg.norm(filtered_matrix, axis=1)  # (N,)
        # Avoid division by zero
        filtered_norms = np.where(filtered_norms == 0, 1.0, filtered_norms)
        similarities = (filtered_matrix @ query_vec.T).squeeze() / (filtered_norms * query_norm)  # (N,)
        
        # Ensure similarities is 1-D even for single result
        similarities = np.atleast_1d(similarities)
        
        # Filter by min_score and build results
        results = []
        for i, (turn, sim) in enumerate(zip(valid_turns, similarities)):
            if sim >= min_score:
                results.append((turn, float(sim)))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def update(
        self,
        memory_id: str,
        content: str,
        dia_id: str,
        session_id: Optional[int] = None,
        session_time: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Update the content of a memory item by its memory ID.
        
        Args:
            memory_id: The memory_id to update
            content: New content to replace the existing content
            dia_id: Dialogue ID (e.g., "D5:4") to append to dia_ids array
            session_id: Optional session id where this update happened
            session_time: Optional session timestamp where this update happened
            
        Returns:
            Updated turn dict if found, None if not found
        """
        for i, turn in enumerate(self.memories):
            if turn["memory_id"] == memory_id:
                # Update content tokens count
                old_content = turn["content"]
                self.total_tokens -= len(self._tokenize(old_content))
                self.total_tokens += len(self._tokenize(content))
                
                # Update content
                turn["content"] = content
                
                # Append dia_id to list (avoid duplicates)
                if "dia_ids" not in turn:
                    turn["dia_ids"] = []
                if dia_id not in turn["dia_ids"]:
                    turn["dia_ids"].append(dia_id)

                # Ensure temporal provenance fields exist even for old rows.
                turn.setdefault("first_session_id", turn.get("session_id"))
                turn.setdefault("first_session_time", turn.get("session_time"))
                turn.setdefault("last_session_id", turn.get("session_id"))
                turn.setdefault("last_session_time", turn.get("session_time"))
                if "mention_history" not in turn:
                    turn["mention_history"] = []

                # When current session metadata is available, record update provenance
                # and advance last-seen temporal fields.
                if session_id is not None and session_time is not None:
                    last_sid = turn.get("last_session_id")
                    if not isinstance(last_sid, int) or session_id >= last_sid:
                        turn["last_session_id"] = session_id
                        turn["last_session_time"] = session_time

                    mention_key = (dia_id, session_id, session_time)
                    seen_mentions = {
                        (m.get("dia_id"), m.get("session_id"), m.get("session_time"))
                        for m in turn["mention_history"]
                        if isinstance(m, dict)
                    }
                    if mention_key not in seen_mentions:
                        turn["mention_history"].append({
                            "dia_id": dia_id,
                            "session_id": session_id,
                            "session_time": session_time,
                        })
                        if len(turn["mention_history"]) > 20:
                            turn["mention_history"] = turn["mention_history"][-20:]
                
                # Track dia_id in the set for easy evaluation
                self.dia_ids_set.add(dia_id)
                
                # Regenerate embedding for the new content
                if memory_id in self._embedding_id_to_idx:
                    idx = self._embedding_id_to_idx[memory_id]
                    new_embedding = self._get_embedding(content, method=self.embedding_method)
                    self.embedding_matrix[idx] = new_embedding
                
                return turn
        return None
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory item by its memory ID.
        
        Args:
            memory_id: The memory_id to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, turn in enumerate(self.memories):
            if turn["memory_id"] == memory_id:
                # Reduce total token count
                self.total_tokens -= len(self._tokenize(turn["content"]))
                
                # Remove dia_ids from the set before deleting the memory
                for dia_id in turn.get('dia_ids', []):
                    # Only remove if no other memory uses this dia_id
                    if not any(dia_id in mem.get('dia_ids', []) for j, mem in enumerate(self.memories) if j != i):
                        self.dia_ids_set.discard(dia_id)
                
                self.memories.pop(i)
                
                # Remove embedding using O(1) lookup and swap-with-last
                if memory_id in self._embedding_id_to_idx:
                    idx = self._embedding_id_to_idx[memory_id]
                    last_idx = self._embed_count - 1
                    if idx != last_idx:
                        # Swap with last element for O(1) removal
                        self.embedding_matrix[idx] = self.embedding_matrix[last_idx]
                        last_id = self.embedding_ids[last_idx]
                        self.embedding_ids[idx] = last_id
                        self._embedding_id_to_idx[last_id] = idx
                    # Remove last element
                    self._embed_count -= 1
                    self.embedding_ids.pop()
                    del self._embedding_id_to_idx[memory_id]
                
                return True
        return False
    
    # --------------------------------------------------
    # Persistence Operations
    # --------------------------------------------------

    def save(self, save_name: str, directory: str = None, split: str = "train", format: str = "pickle") -> str:
        """
        Save all current memories to disk with a given name.
        Saves memories, embeddings, and metadata in a single location.
        
        Note: Both pickle and JSON formats are always saved:
        - .pkl: Contains memories + embeddings for fast loading
        - .json: Contains only memories for human readability
        - _metadata.json: Contains save metadata
        
        Args:
            save_name: Name for this memory save (e.g., "conv-41_session-5")
            directory: Base directory to save memories (default: "memory_store")
            format: Primary format indicator - "pickle" (default) or "json" (affects return message only)
            
        Returns:
            Path to the saved file (without extension)
        """
        if directory is None:
            # Check for MEMORY_CACHE_DIR environment variable
            if split == "train":
                directory = os.environ.get('MEMORY_CACHE_DIR')
            elif split == "validation":
                directory = os.environ.get('MEMORY_CACHE_DIR_VAL')
            elif split == "test":
                directory = os.environ.get('MEMORY_CACHE_DIR_TEST')

            if directory is None:
                # Use current directory + '/memory_store'
                directory = os.path.join(os.getcwd(), 'memory_store')
        base_path = Path(directory)
        base_path.mkdir(parents=True, exist_ok=True)
        
        save_path = base_path / save_name
        
        if format == "pickle":
            # Save everything in pickle: memories (including dia_ids) + embeddings for fast loading
            # This makes snapshots self-contained and avoids cache lookups during load
            import pickle
            with open(f"{save_path}.pkl", "wb") as f:
                pickle.dump({
                    'memories': self.memories,  # Each memory contains dia_ids array
                    'embedding_matrix': self.embedding_matrix[:self._embed_count],  # Only save used portion
                    'embedding_ids': self.embedding_ids,
                    'dia_ids_set': self.dia_ids_set  # Set of all dia_ids for easy evaluation
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # JSON format: save memories + metadata for human readability
            with open(f"{save_path}.json", "w") as f:
                json.dump(self.memories, f, indent=2)
            
            metadata = {
                "save_name": save_name,
                "total_memories": len(self.memories),
                "unique_conversations": len(set(m["sample_id"] for m in self.memories)) if self.memories else 0,
                "unique_speakers": list(set(m["speaker"] for m in self.memories)) if self.memories else [],
                "saved_at": datetime.now().isoformat(),
                "embedding_method": self.embedding_method,
                "format": format
            }
            with open(f"{save_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        else:
            # JSON format: save memories + metadata for human readability
            with open(f"{save_path}.json", "w") as f:
                json.dump(self.memories, f, indent=2)
            
            metadata = {
                "save_name": save_name,
                "total_memories": len(self.memories),
                "unique_conversations": len(set(m["sample_id"] for m in self.memories)) if self.memories else 0,
                "unique_speakers": list(set(m["speaker"] for m in self.memories)) if self.memories else [],
                "saved_at": datetime.now().isoformat(),
                "embedding_method": self.embedding_method,
                "format": format
            }
            with open(f"{save_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        ext = ".pkl" if format == "pickle" else ".json"
        # print(f"✓ Saved {len(self.memories)} memories to '{save_name}{ext}' in {directory}")
        return str(save_path)
    
    def load(self, save_name: str, directory: str = "memory_store", 
             clear_existing: bool = True, format: str = None) -> int:
        """
        Load memories from disk.
        
        Args:
            save_name: Name of the saved memory to load
            directory: Base directory containing saved memories
            clear_existing: If True, clear current memories before loading (default: True)
            format: Serialization format - "pickle", "json", or None (auto-detect)
            
        Returns:
            Number of memories loaded
        """
        base_path = Path(directory)
        save_path = base_path / save_name
        
        # Auto-detect format if not specified
        if format is None:
            pkl_file = Path(f"{save_path}.pkl")
            json_file = Path(f"{save_path}.json")
            
            if pkl_file.exists():
                format = "pickle"
            elif json_file.exists():
                format = "json"
            else:
                raise FileNotFoundError(f"Memory file not found: {save_path}.pkl or {save_path}.json")
        
        if clear_existing:
            self.memories = []
            self._embed_capacity = _INITIAL_EMBED_CAPACITY
            self.embedding_matrix = np.empty((self._embed_capacity, self._embedding_dim))
            self._embed_count = 0
            self.embedding_ids = []
            self._embedding_id_to_idx = {}
            self.dia_ids_set = set()
            self.total_tokens = 0
        
        # Load memories based on format
        if format == "pickle":
            import pickle
            memories_file = f"{save_path}.pkl"
            if not Path(memories_file).exists():
                raise FileNotFoundError(f"Memory file not found: {memories_file}")
            
            with open(memories_file, "rb") as f:
                data = pickle.load(f)
            
            # Pickle format always contains dict with embeddings
            loaded_memories = data['memories']
            loaded_embedding_matrix = data.get('embedding_matrix')
            loaded_embedding_ids = data.get('embedding_ids')
            loaded_dia_ids_set = data.get('dia_ids_set', set())  # For backward compatibility
        else:
            # JSON format: only memories, no embeddings
            memories_file = f"{save_path}.json"
            if not Path(memories_file).exists():
                raise FileNotFoundError(f"Memory file not found: {memories_file}")
            
            with open(memories_file, "r") as f:
                loaded_memories = json.load(f)
            loaded_embedding_matrix = None
            loaded_embedding_ids = None
        
        # Add to current memories (check for duplicates by memory_id if not clearing)
        if clear_existing:
            self.memories = loaded_memories
            loaded_count = len(loaded_memories)
            
            # Recalculate total tokens
            self.total_tokens = sum(len(self._tokenize(m["content"])) for m in self.memories)
            
            # Restore embeddings if available (pickle new format)
            if loaded_embedding_matrix is not None:
                self._embed_count = len(loaded_embedding_ids)
                self._embed_capacity = max(_INITIAL_EMBED_CAPACITY, self._embed_count * 2)
                self.embedding_matrix = np.empty((self._embed_capacity, self._embedding_dim))
                self.embedding_matrix[:self._embed_count] = loaded_embedding_matrix
                self.embedding_ids = loaded_embedding_ids
                self._embedding_id_to_idx = {mid: i for i, mid in enumerate(self.embedding_ids)}
            else:
                # No embeddings saved, need to rebuild from cache
                if self.memories:
                    self._rebuild_embeddings()
            
            # Restore dia_ids_set (for pickle format)
            if format == "pickle" and loaded_dia_ids_set:
                # Use loaded dia_ids_set if it exists and is not empty
                self.dia_ids_set = loaded_dia_ids_set
            else:
                # Rebuild dia_ids_set from loaded memories (for old caches or JSON format)
                self.dia_ids_set = set()
                for memory in self.memories:
                    for dia_id in memory.get('dia_ids', []):
                        self.dia_ids_set.add(dia_id)
                # print(f"✓ Rebuilt dia_ids_set from memories: {len(self.dia_ids_set)} unique dia_ids")
        else:
            existing_ids = set(m["memory_id"] for m in self.memories)
            loaded_count = 0
            for memory in loaded_memories:
                if memory["memory_id"] not in existing_ids:
                    self.memories.append(memory)
                    self.total_tokens += len(self._tokenize(memory["content"]))
                    loaded_count += 1
            
            # For append mode, always rebuild embeddings for new memories
            if loaded_count > 0:
                existing_ids_set = set(self.embedding_ids)
                for memory in self.memories:
                    mid = memory["memory_id"]
                    if mid not in existing_ids_set:
                        embedding = self._get_embedding(memory["content"], method=self.embedding_method)
                        if self._embed_count >= self._embed_capacity:
                            self._grow_embedding_matrix()
                        self.embedding_matrix[self._embed_count] = embedding
                        self.embedding_ids.append(mid)
                        self._embedding_id_to_idx[mid] = self._embed_count
                        self._embed_count += 1
                        # Also add dia_ids to the set
                        for dia_id in memory.get('dia_ids', []):
                            self.dia_ids_set.add(dia_id)
        
        # print(f"✓ Loaded {loaded_count} memories from '{save_name}' in {directory}")
        return loaded_count
    
    def _grow_embedding_matrix(self) -> None:
        """Double the capacity of the pre-allocated embedding matrix."""
        new_capacity = self._embed_capacity * 2
        new_matrix = np.empty((new_capacity, self._embedding_dim))
        new_matrix[:self._embed_count] = self.embedding_matrix[:self._embed_count]
        self.embedding_matrix = new_matrix
        self._embed_capacity = new_capacity

    def _rebuild_embeddings(self) -> None:
        """Rebuild the embedding matrix from current memories (uses batch API + cache)."""
        count = len(self.memories)
        self._embed_capacity = max(_INITIAL_EMBED_CAPACITY, count * 2)
        self.embedding_matrix = np.empty((self._embed_capacity, self._embedding_dim))
        self._embed_count = 0
        self.embedding_ids = []
        self._embedding_id_to_idx = {}
        
        if not self.memories:
            return
        
        # Batch fetch all embeddings at once
        texts = [memory["content"] for memory in self.memories]
        embeddings = self._get_embeddings_batch(texts, method=self.embedding_method)
        
        for memory, embedding in zip(self.memories, embeddings):
            self.embedding_matrix[self._embed_count] = embedding
            mid = memory["memory_id"]
            self.embedding_ids.append(mid)
            self._embedding_id_to_idx[mid] = self._embed_count
            self._embed_count += 1
    
    def list_saves(self, directory: str = "memory_store") -> List[Dict[str, any]]:
        """
        List all saved memory files in the directory.
        
        Args:
            directory: Base directory containing saved memories
            
        Returns:
            List of metadata dicts for each saved memory
        """
        base_path = Path(directory)
        if not base_path.exists():
            return []
        
        saves = []
        # Find all JSON files (excluding metadata files)
        for json_file in base_path.glob("*.json"):
            if json_file.name.endswith("_metadata.json"):
                # Read metadata file
                with open(json_file, "r") as f:
                    metadata = json.load(f)
                    metadata["path"] = str(json_file.parent / metadata["save_name"])
                    saves.append(metadata)
        
        return sorted(saves, key=lambda x: x.get("saved_at", ""), reverse=True)
    
