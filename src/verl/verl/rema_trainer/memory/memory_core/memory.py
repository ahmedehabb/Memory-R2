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
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from verl.rema_trainer.memory.memory_core.embedding_cache import get_cache

class Memory:
    """Stores conversation turns with metadata in RAM."""

    EMBEDDING_METHOD = "openai"

    def __init__(self, embedding_method: str = None, enable_cache: bool = True, cache_dir: str = None) -> None:
        # Single memory store: conversation turns with metadata
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
        
        # Embeddings stored as matrices for batch operations
        self.embedding_matrix: np.ndarray = np.empty((0, self._embedding_dim))
        # Memory ID mappings to track which row corresponds to which memory
        self.embedding_ids: List[str] = []  # memory_ids

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
                load_dotenv()
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
        Insert a conversation turn with metadata.
        
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
                # Return existing memory instead of creating duplicate
                return existing_memory
        
        memory_id = self._generate_memory_id()
        
        turn_data = {
            "memory_id": memory_id,
            "sample_id": sample_id,
            "session_id": session_id,
            "session_time": session_time,
            "speaker": speaker,
            "content": content,
            "dia_ids": [dia_id]  # Store dia_id in array
        }
        
        self.memories.append(turn_data)
        
        # Generate and store embedding for the content
        embedding = self._get_embedding(content, method=self.embedding_method)
        new_matrix = np.vstack([self.embedding_matrix, embedding.reshape(1, -1)])
        self.embedding_matrix = new_matrix
        self.embedding_ids.append(memory_id)
        
        return turn_data
    
    def get(self, sample_id: str = None, speaker: str = None) -> List[Dict[str, any]]:
        """
        Retrieve conversation turns, optionally filtered by sample_id and/or speaker.
        
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
        Search conversation turns using semantic similarity or BM25.
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
        
        if not filtered_turns or not query.strip():
            return []
        
        if search_method == "text-embedding":
            return self._search_embedding(filtered_turns, query, top_k, min_score)
        elif search_method == "bm25":
            return self._search_bm25(filtered_turns, query, top_k, min_score)
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'bm25' or 'text-embedding'.")
    
    def _search_embedding(self, turns: List[Dict[str, any]], query: str,
                          top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, any], float]]:
        """Search conversation turns using embedding similarity."""
        if not turns:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query, method=self.embedding_method)
        if np.allclose(query_embedding, 0):
            return []
        
        results = []
        
        # Get embeddings for the filtered turns
        for turn in turns:
            memory_id = turn["memory_id"]
            if memory_id in self.embedding_ids:
                idx = self.embedding_ids.index(memory_id)
                turn_embedding = self.embedding_matrix[idx]
                
                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    turn_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= min_score:
                    results.append((turn, float(similarity)))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def _search_bm25(self, turns: List[Dict[str, any]], query: str,
                     top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, any], float]]:
        """Search conversation turns using BM25."""
        if not turns:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Tokenize all turn contents
        tokenized_corpus = []
        for turn in turns:
            doc_tokens = self._tokenize(turn["content"])
            tokenized_corpus.append(doc_tokens)
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Get scores
        doc_scores = bm25.get_scores(query_tokens)
        
        # Create results
        results = []
        for i, turn in enumerate(turns):
            score = doc_scores[i]
            if score >= min_score:
                results.append((turn, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def update(self, memory_id: str, content: str, dia_id: str) -> Dict[str, any]:
        """
        Update the content of a conversation turn by its memory ID.
        
        Args:
            memory_id: The memory_id to update
            content: New content to replace the existing content
            dia_id: Dialogue ID (e.g., "D5:4") to append to dia_ids array
            
        Returns:
            Updated turn dict if found, None if not found
        """
        for i, turn in enumerate(self.memories):
            if turn["memory_id"] == memory_id:
                # Update content
                turn["content"] = content
                
                # Append dia_id to list (avoid duplicates)
                if "dia_ids" not in turn:
                    turn["dia_ids"] = []
                if dia_id not in turn["dia_ids"]:
                    turn["dia_ids"].append(dia_id)
                
                # Regenerate embedding for the new content
                if memory_id in self.embedding_ids:
                    idx = self.embedding_ids.index(memory_id)
                    new_embedding = self._get_embedding(content, method=self.embedding_method)
                    self.embedding_matrix[idx] = new_embedding
                
                return turn
        return None
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a conversation turn by its memory ID.
        
        Args:
            memory_id: The memory_id to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, turn in enumerate(self.memories):
            if turn["memory_id"] == memory_id:
                self.memories.pop(i)
                
                # Remove embedding
                if memory_id in self.embedding_ids:
                    idx = self.embedding_ids.index(memory_id)
                    new_matrix = np.delete(self.embedding_matrix, idx, axis=0)
                    self.embedding_matrix = new_matrix
                    self.embedding_ids.pop(idx)
                
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
            # Save everything in pickle: memories + embeddings for fast loading
            # This makes snapshots self-contained and avoids cache lookups during load
            import pickle
            with open(f"{save_path}.pkl", "wb") as f:
                pickle.dump({
                    'memories': self.memories,
                    'embedding_matrix': self.embedding_matrix,
                    'embedding_ids': self.embedding_ids
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Always save JSON for human readability (regardless of format)
        with open(f"{save_path}.json", "w") as f:
            json.dump(self.memories, f, indent=2)
        
        # Save metadata (always JSON for human readability)
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
        print(f"✓ Saved {len(self.memories)} memories to '{save_name}{ext}' in {directory}")
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
            self.embedding_matrix = np.empty((0, self._embedding_dim))
            self.embedding_ids = []
        
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
            
            # Restore embeddings if available (pickle new format)
            if loaded_embedding_matrix is not None:
                self.embedding_matrix = loaded_embedding_matrix
                self.embedding_ids = loaded_embedding_ids
            else:
                # No embeddings saved, need to rebuild from cache
                if self.memories:
                    self._rebuild_embeddings()
        else:
            existing_ids = set(m["memory_id"] for m in self.memories)
            loaded_count = 0
            for memory in loaded_memories:
                if memory["memory_id"] not in existing_ids:
                    self.memories.append(memory)
                    loaded_count += 1
            
            # For append mode, always rebuild embeddings for new memories
            if loaded_count > 0:
                existing_ids_set = set(self.embedding_ids)
                for memory in self.memories:
                    mid = memory["memory_id"]
                    if mid not in existing_ids_set:
                        embedding = self._get_embedding(memory["content"], method=self.embedding_method)
                        self.embedding_matrix = np.vstack([self.embedding_matrix, embedding.reshape(1, -1)])
                        self.embedding_ids.append(mid)
        
        print(f"✓ Loaded {loaded_count} memories from '{save_name}' in {directory}")
        return loaded_count
    
    def _rebuild_embeddings(self) -> None:
        """Rebuild the embedding matrix from current memories (uses cache)."""
        self.embedding_matrix = np.empty((0, self._embedding_dim))
        self.embedding_ids = []
        
        for memory in self.memories:
            embedding = self._get_embedding(memory["content"], method=self.embedding_method)
            self.embedding_matrix = np.vstack([self.embedding_matrix, embedding.reshape(1, -1)])
            self.embedding_ids.append(memory["memory_id"])
    
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
    
