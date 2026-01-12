"""
Memory Manager - Handles LLM-generated JSON commands and executes memory actions.

This module provides a high-level interface for managing memory operations
based on JSON commands from an LLM. It validates, parses, and executes
memory operations safely.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import json
import os
from pathlib import Path
from verl.rema_trainer.memory.memory_core.memory import Memory


class MemoryManager:
    """
    Manages memory operations based on LLM-generated JSON commands.
    
    Supported operations:
    - insert: Add new conversation turns
    - update: Update existing conversation turn content
    - search: Find relevant conversation turns
    - delete: Remove conversation turns by ID
    
    Note: This manager is stateless - memory instances are passed to methods.
    This allows for more flexible usage patterns (e.g., managing multiple conversations).
    """
    
    def __init__(self, embedding_method: str = "openai", enable_cache: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            embedding_method: Default embedding method for creating new memories
            enable_cache: Whether to enable embedding cache by default
        """
        self.embedding_method = embedding_method
        self.enable_cache = enable_cache
        self.operation_history: List[Dict[str, Any]] = []
    
    def execute_command(self, memory: Memory, command: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a single memory command on the given memory instance.
        
        Args:
            memory: Memory instance to operate on
            command: JSON string or dict with memory operation
            
        Returns:
            Dict with execution result including status, message, and data
            
        Example command formats:
            {
                "operation": "insert",
                "sample_id": "conv-41",
                "session_id": 1,
                "session_time": "11:01 am on 17 December, 2022",
                "speaker": "John",
                "content": "LLM-generated summary of the conversation turn",
                "dia_id": "D1:4"
            }
            
            {
                "operation": "delete",
                "memory_id": "a1b2c3d4"
            }
            
            {
                "operation": "update",
                "memory_id": "a1b2c3d4",
                "content": "Updated summary of the conversation turn",
                "dia_id": "D1:4"
            }
            
            {
                "operation": "search",
                "query": "programming languages",
                "sample_id": "conv-41",  # Optional filter
                "speaker": "John",  # Optional filter
                "top_k": 5,
                "search_method": "text-embedding"
            }
        """
        try:
            # Parse JSON if string
            if isinstance(command, str):
                command = json.loads(command)
            
            # Validate command structure
            if not isinstance(command, dict):
                return self._error_result("Command must be a JSON object")
            
            operation = command.get("operation")
            if not operation:
                return self._error_result("Missing 'operation' field")
            
            # Normalize operation to lowercase for consistency
            operation = operation.lower()
            
            # Route to appropriate handler
            if operation == "insert":
                result = self._handle_insert(memory, command)
            elif operation == "update":
                result = self._handle_update(memory, command)
            elif operation == "delete":
                result = self._handle_delete(memory, command)
            elif operation == "search":
                result = self._handle_search(memory, command)
            else:
                result = self._error_result(f"Unknown operation: {operation}")
            
            # Log successful operations
            if result["status"] == "success":
                self.operation_history.append({
                    "command": command,
                    "result": result
                })
            
            return result
            
        except json.JSONDecodeError as e:
            return self._error_result(f"Invalid JSON: {e}")
        except Exception as e:
            return self._error_result(f"Execution error: {e}")
    
    def execute_batch(self, memory: Memory, commands: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Execute multiple memory commands in sequence on the given memory instance.
        
        Args:
            memory: Memory instance to operate on
            commands: JSON string (array) or list of command dicts
            
        Returns:
            Dict with batch execution results
            
        Example:
            [
                {"operation": "insert", "sample_id": "conv-41", "session_id": 1, "session_time": "...", "speaker": "John", "content": "Summary"},
                {"operation": "search", "query": "summary", "top_k": 5}
            ]
        """
        try:
            # Parse JSON if string
            if isinstance(commands, str):
                commands = json.loads(commands)
            
            if not isinstance(commands, list):
                return self._error_result("Batch commands must be a JSON array")
            
            results = []
            success_count = 0
            error_count = 0
            
            for i, command in enumerate(commands):
                result = self.execute_command(memory, command)
                results.append({
                    "command_index": i,
                    "command": command,
                    "result": result
                })
                
                if result["status"] == "success":
                    success_count += 1
                else:
                    error_count += 1
            
            return {
                "status": "success" if error_count == 0 else "partial",
                "total_commands": len(commands),
                "successful": success_count,
                "failed": error_count,
                "results": results
            }
            
        except json.JSONDecodeError as e:
            return self._error_result(f"Invalid JSON: {e}")
        except Exception as e:
            return self._error_result(f"Batch execution error: {e}")
    
    def _handle_insert(self, memory: Memory, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle insert operation."""
        # Required fields for conversation turn
        sample_id = command.get("sample_id")
        session_id = command.get("session_id")
        session_time = command.get("session_time")
        speaker = command.get("speaker")
        content = command.get("content")
        dia_id = command.get("dia_id")
        
        # Validate required fields
        if not sample_id:
            return self._error_result("Missing 'sample_id' field")
        if session_id is None:
            return self._error_result("Missing 'session_id' field")
        if not session_time:
            return self._error_result("Missing 'session_time' field")
        if not speaker:
            return self._error_result("Missing 'speaker' field")
        if not content:
            return self._error_result("Missing 'content' field")
        if not dia_id:
            return self._error_result("Missing 'dia_id' field")
        
        # Validate dia_id is a string, not a list
        if isinstance(dia_id, list):
            return self._error_result(f"dia_id must be a string, not a list. Got: {dia_id}")
        
        try:
            turn_data = memory.insert(
                sample_id=sample_id,
                session_id=session_id,
                session_time=session_time,
                speaker=speaker,
                content=content,
                dia_id=dia_id
            )
            
            return {
                "status": "success",
                "operation": "insert",
                "memory_id": turn_data["memory_id"],
                "message": "Conversation turn inserted successfully",
                "data": turn_data
            }
        except Exception as e:
            return self._error_result(f"Insert failed: {e}")
    
    def _handle_update(self, memory: Memory, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update operation."""
        memory_id = command.get("memory_id")
        content = command.get("content")
        dia_id = command.get("dia_id")
        
        # Validate required fields
        if not memory_id:
            return self._error_result("Missing 'memory_id' field")
        if not content:
            return self._error_result("Missing 'content' field")
        if not dia_id:
            return self._error_result("Missing 'dia_id' field")
        
        # Validate dia_id is a string, not a list
        if isinstance(dia_id, list):
            return self._error_result(f"dia_id must be a string, not a list. Got: {dia_id}")
        
        try:
            updated_turn = memory.update(memory_id, content, dia_id)
            
            if updated_turn:
                return {
                    "status": "success",
                    "operation": "update",
                    "memory_id": memory_id,
                    "message": "Turn updated successfully",
                    "data": updated_turn
                }
            else:
                return self._error_result(f"memory_id '{memory_id}' not found: current memory ids are {[t['memory_id'] for t in memory.memories]}")
        except Exception as e:
            return self._error_result(f"Update failed: {e}")
    
    def _handle_delete(self, memory: Memory, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete operation."""
        memory_id = command.get("memory_id")
        
        # Validate required fields
        if not memory_id:
            return self._error_result("Missing 'memory_id' field")
        
        try:
            success = memory.delete(memory_id)
            
            if success:
                return {
                    "status": "success",
                    "operation": "delete",
                    "memory_id": memory_id,
                    "message": "Turn deleted successfully",
                    "data": None
                }
            else:
                return self._error_result(f"memory_id '{memory_id}' not found: current memory ids are {[t['memory_id'] for t in memory.memories]}")
        except Exception as e:
            return self._error_result(f"Delete failed: {e}")
    
    def _handle_search(self, memory: Memory, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search operation."""
        query = command.get("query")
        sample_id = command.get("sample_id")  # Optional filter
        speaker = command.get("speaker")  # Optional filter
        top_k = command.get("top_k")
        min_score = command.get("min_score", 0.0)
        search_method = command.get("search_method", "text-embedding")
        
        # Validate required fields
        if not query:
            return self._error_result("Missing 'query' field")
        
        # Validate search method
        if search_method not in ["bm25", "text-embedding"]:
            return self._error_result(f"Invalid search_method: {search_method}")
        
        try:
            results = memory.search(
                query=query,
                sample_id=sample_id,
                speaker=speaker,
                top_k=top_k,
                min_score=min_score,
                search_method=search_method
            )
            
            # Format results
            formatted_results = []
            for turn, score in results:
                formatted_results.append({
                    "memory_id": turn["memory_id"],
                    "sample_id": turn["sample_id"],
                    "session_id": turn["session_id"],
                    "session_time": turn["session_time"],
                    "speaker": turn["speaker"],
                    "content": turn["content"],
                    "score": float(score)
                })
            
            return {
                "status": "success",
                "operation": "search",
                "query": query,
                "search_method": search_method,
                "message": f"Found {len(formatted_results)} turns",
                "data": formatted_results
            }
        except Exception as e:
            return self._error_result(f"Search failed: {e}")
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create a standardized error result."""
        return {
            "status": "error",
            "message": message,
            "data": None
        }
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the history of successful operations.
        
        Args:
            limit: Optional limit on number of operations to return (most recent first)
            
        Returns:
            List of operation history entries
        """
        if limit is None:
            return self.operation_history
        else:
            return self.operation_history[-limit:]
    
    def clear_history(self):
        """Clear the operation history."""
        self.operation_history = []
    
    def get_memory_stats(self, memory: Memory) -> Dict[str, Any]:
        """
        Get statistics about the given memory state.
        
        Args:
            memory: Memory instance to get stats for
            
        Returns:
            Dict with memory statistics
        """
        return {
            "total_turns": len(memory.memories),
            "unique_conversations": len(set(t["sample_id"] for t in memory.memories)),
            "unique_speakers": len(set(t["speaker"] for t in memory.memories)),
            "total_operations": len(self.operation_history),
        }

    def get_snapshot(self, sample_id: str, chunk_id: int, epoch: int, split: str = "train", index_in_batch: int = -1) -> Optional[Memory]:
        """
        Get a memory snapshot for a specific conversation, chunk, and epoch.
        
        This loads a previously saved memory state from disk. If no snapshot exists,
        returns None (caller should create a new Memory instance).
        Auto-detects pickle or JSON format (pickle is preferred for speed).
        
        Args:
            sample_id: Conversation ID (e.g., "conv-41")
            chunk_id: Chunk number (e.g., 0, 1, 2, ...)
            epoch: Training epoch number
            index_in_batch: Optional index in batch to match saved snapshot (default -1 means no index)
            split: Data split name (e.g., "train", "validation")
            
        Returns:
            Memory instance with the loaded state, or None if no snapshot exists
            
        Directory structure: {MEMORY_CACHE_DIR}/epoch_{epoch}/{sample_id}/chunk_{chunk_id}
        
        Example usage:
            manager = MemoryManager()
            
            # Try to load existing snapshot
            memory = manager.get_snapshot("conv-41", chunk_id=1, epoch=2, split="train")
            
            if memory is None:
                # No snapshot exists, create new memory
                memory = Memory(embedding_method="openai")
                print("Created new memory (no snapshot found)")
            else:
                print(f"Loaded snapshot with {len(memory.memories)} memories")
        """
        # Get base directory from environment or use default
        if split == "train":
            base_dir = os.environ.get('MEMORY_CACHE_DIR')
        elif split == "validation":
            base_dir = os.environ.get('MEMORY_CACHE_DIR_VAL')
        elif split == "test":
            base_dir = os.environ.get('MEMORY_CACHE_DIR_TEST')
        
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), 'memory_snapshots')
        
        # Construct directory path: base/epoch_N/conv-ID/
        snapshot_dir = Path(base_dir) / f"epoch_{epoch}" / sample_id
        # Only use index_in_batch suffix for train split (test/validation have no repeated indices)
        if split == "train" and index_in_batch >= 0:
            snapshot_name = f"chunk_{chunk_id}_idx_{index_in_batch}"
        else:
            snapshot_name = f"chunk_{chunk_id}"
        
        # Check for pickle file first (faster), then JSON
        snapshot_pkl = snapshot_dir / f"{snapshot_name}.pkl"
        snapshot_json = snapshot_dir / f"{snapshot_name}.json"
        
        if snapshot_pkl.exists():
            print("Found pickle snapshot")
            # Load pickle format (fast)
            new_memory = Memory(
                embedding_method=self.embedding_method,
                enable_cache=self.enable_cache
            )
            try:
                new_memory.load(snapshot_name, directory=str(snapshot_dir), format="pickle")
                print(f"✓ Loaded snapshot (pickle): epoch_{epoch}/{sample_id}/chunk_{chunk_id} ({len(new_memory.memories)} memories)")
                return new_memory
            except Exception as e:
                print(f"⚠ Error loading pickle snapshot, returning None: {e}")
                return None
        elif snapshot_json.exists():
            # Load JSON format (slower, for backwards compatibility)
            new_memory = Memory(
                embedding_method=self.embedding_method,
                enable_cache=self.enable_cache
            )
            try:
                new_memory.load(snapshot_name, directory=str(snapshot_dir), format="json")
                print(f"✓ Loaded snapshot (json): epoch_{epoch}/{sample_id}/chunk_{chunk_id} ({len(new_memory.memories)} memories)")
                return new_memory
            except Exception as e:
                print(f"⚠ Error loading json snapshot, returning None: {e}")
                return None
        else:
            # No snapshot exists
            if index_in_batch >= 0:
                print(f"ℹ No snapshot found at epoch_{epoch}/{sample_id}/chunk_{chunk_id}_idx_{index_in_batch}")
            else:
                print(f"ℹ No snapshot found at epoch_{epoch}/{sample_id}/chunk_{chunk_id}")
            return None

    def cache_snapshot(self, memory: Memory, sample_id: str, chunk_id: int, epoch: int, split: str = "train", index_in_batch: int = -1, format: str = "pickle") -> str:
        """
        Cache the given memory state as a snapshot for a specific conversation, chunk, and epoch.
        
        This saves the memory state to disk so it can be loaded later.
        Uses pickle format by default for ~5-10x faster loading compared to JSON.
        
        Args:
            memory: Memory instance to save
            sample_id: Conversation ID (e.g., "conv-41")
            chunk_id: Chunk number (e.g., 0, 1, 2, ...)
            epoch: Training epoch number
            index_in_batch: Optional index in batch to avoid overwriting (default -1 means no index)
            split: Data split name (e.g., "train", "validation")
            format: Serialization format - "pickle" (fast, default) or "json" (human-readable)
            
        Returns:
            Path to the saved snapshot
            
        Directory structure: {MEMORY_CACHE_DIR}/epoch_{epoch}/{sample_id}/chunk_{chunk_id}
        
        Example usage:
            manager = MemoryManager()
            memory = Memory(embedding_method="openai")
            
            # ... add some memories ...
            memory.insert("conv-41", 1, "10:00 am", "John", "Likes hiking")
            
            # Save snapshot (uses pickle by default)
            path = manager.cache_snapshot(memory, "conv-41", chunk_id=0, epoch=1)
            print(f"Saved to: {path}")
        """
        # Get base directory from environment or use default
        if split == "train":
            base_dir = os.environ.get('MEMORY_CACHE_DIR')
        elif split == "validation":
            base_dir = os.environ.get('MEMORY_CACHE_DIR_VAL')
        elif split == "test":
            base_dir = os.environ.get('MEMORY_CACHE_DIR_TEST')

        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), 'memory_snapshots')
        
        # Construct directory path: base/epoch_N/conv-ID/
        snapshot_dir = Path(base_dir) / f"epoch_{epoch}" / sample_id
        # Only use index_in_batch suffix for train split (test/validation have no repeated indices)
        if split == "train" and index_in_batch >= 0:
            snapshot_name = f"chunk_{chunk_id}_idx_{index_in_batch}"
        else:
            snapshot_name = f"chunk_{chunk_id}"
        
        # Save snapshot with specified format
        save_path = memory.save(snapshot_name, directory=str(snapshot_dir), format=format)
        ext = ".pkl" if format == "pickle" else ".json"
        print(f"✓ Cached snapshot: epoch_{epoch}/{sample_id}/{snapshot_name}{ext} ({len(memory.memories)} memories)")
        
        return save_path

    def attach_turn_metadata_to_operations(self, operations: List[Dict[str, Any]], turns: List[Dict[str, Any]], conv_id: str) -> List[Dict[str, Any]]:
        """
        Attach metadata from conversation turns to insert operations.

        **IMPORTANT CONSTRAINT**: All turns in the input list MUST belong to the same 
        conversation (same sample_id) AND the same session (same session_id, session_time).
        This ensures that memory entries are tagged with accurate temporal metadata.

        This function enriches LLM-generated operations with necessary metadata fields
        (sample_id, session_id, session_time) by extracting them from the conversation turns.
        Since all turns share the same metadata, we use the first turn as the source.

        Args:
            operations: List of operation dicts from LLM
            turns: List of conversation turn dicts from a SINGLE SESSION. All turns must have:
                - Same session_id (session number)
                - Same session_time (timestamp)
                Example turn: {"session_id": 9, "session_time": "6:59 pm on 26 August, 2023", "speaker": "John", ...}
            conv_id: The conversation ID (sample_id) that all turns belong to

        Returns:
            The operations list with 'sample_id', 'session_id', and 'session_time' filled
            on any operation with operation == "insert" (only if those keys are missing).
        """
        if not turns:
            return operations

        if not operations:
            return []

        sample_id = conv_id
        # All turns belong to the same conversation and session, so we can use the first turn's metadata
        first_turn = turns[0]
        session_id = first_turn.get("session_id")
        session_time = first_turn.get("session_time")

        for op in operations:
            # Skip if operation is not a dict (malformed)
            if not isinstance(op, dict):
                continue
            
            # Normalize operation to lowercase for consistency
            # Handle None values explicitly
            operation_value = op.get("operation") or ""
            op["operation"] = operation_value.lower()
            if op["operation"] == "insert":
                # Only set metadata when it's not already provided
                if sample_id is not None:
                    op.setdefault("sample_id", sample_id)
                if session_id is not None:
                    op.setdefault("session_id", session_id)
                if session_time is not None:
                    op.setdefault("session_time", session_time)

        return operations

def create_function_schema() -> List[Dict[str, Any]]:
    """
    Create OpenAI function calling schema for memory operations.
    
    Returns:
        List of function schemas compatible with OpenAI's function calling API
    """
    return [
        {
            "name": "insert_memory",
            "description": "Insert a new conversation turn with metadata into memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "Conversation ID (e.g., 'conv-41')"
                    },
                    "session_id": {
                        "type": "integer",
                        "description": "Session number within the conversation"
                    },
                    "session_time": {
                        "type": "string",
                        "description": "Timestamp of the session (e.g., '11:01 am on 17 December, 2022')"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Name of the speaker"
                    },
                    "content": {
                        "type": "string",
                        "description": "LLM-generated summary/content to save"
                    }
                },
                "required": ["sample_id", "session_id", "session_time", "speaker", "content"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search for conversation turns using semantic similarity or keyword matching",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "sample_id": {
                        "type": "string",
                        "description": "Optional: Filter by conversation ID"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Optional: Filter by speaker name"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0.0-1.0)"
                    },
                    "search_method": {
                        "type": "string",
                        "enum": ["bm25", "text-embedding"],
                        "description": "Search method: 'bm25' for keyword, 'text-embedding' for semantic"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "update_memory",
            "description": "Update the content of an existing conversation turn by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the turn to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content to replace the existing content"
                    }
                },
                "required": ["memory_id", "content"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a conversation turn by its ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the turn to delete"
                    }
                },
                "required": ["memory_id"]
            }
        }
    ]
