"""
QA prompt generator for question answering with memory.

This module generates prompts for the LLM to answer questions based on
memories from two speakers in a conversation.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from verl.rema_trainer.memory.memory_core.memory import Memory


def format_memories_for_speaker(
    memory: Memory,
    speaker_name: str,
    query: str = None,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True
) -> List[Dict[str, Any]]:
    """
    Format memories for a specific speaker, optionally using similarity search.
    
    Args:
        memory: Memory instance containing all memories
        speaker_name: Name of the speaker to filter memories for
        query: Query string for similarity search (optional)
        top_k: Maximum number of relevant memories to retrieve (default: 5)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, return all speaker memories (default: True)
        
    Returns:
        List of formatted memory dicts for the speaker without memory_id.
        
    Example output: 
        [
            {
                "session_time": "5:00 pm on 26 August, 2023",
                "content": "Enjoys reading books from various genres"
            }
        ]
    """
    formatted_memories = []
    
    if use_similarity and query:
        # Use memory's search function to find relevant memories
        # Returns List[Tuple[Dict, float]] where tuple is (memory_dict, similarity_score)
        search_results = memory.search(
            query=query,
            speaker=speaker_name,  # Filter by speaker
            top_k=top_k,
            search_method="text-embedding"
        )
        
        # Filter by similarity threshold, deduplicate by content, and format
        seen_contents = set()
        for memory_dict, similarity_score in search_results:
            if similarity_score >= similarity_threshold:
                content = memory_dict.get("content")
                # Deduplicate by content
                if content not in seen_contents:
                    seen_contents.add(content)
                    formatted_memories.append({
                        # "memory_id": memory_dict.get("memory_id"), # Removed memory_id from output
                        "session_time": memory_dict.get("session_time"),
                        "content": content
                    })
    else:
        # Fallback: return all memories for this speaker (original behavior)
        all_memories = memory.get()
        speaker_memories = [mem for mem in all_memories if mem.get("speaker") == speaker_name]
        
        # Format the memories
        for mem in speaker_memories:
            formatted_memories.append({
                # "memory_id": mem.get("memory_id"), # Removed memory_id from output
                "session_time": mem.get("session_time"),
                "content": mem.get("content")
            })
    
    return formatted_memories


def generate_qa_prompt(
    memory: Memory,
    speaker_1: str,
    speaker_2: str,
    question: str,
    session_time: str,
    top_k_per_speaker: int = 5,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True,
    prompt_template_path: Optional[str] = None
) -> str:
    """
    Generate the complete prompt for question answering based on memories.
    
    Uses similarity search to retrieve only the most relevant memories for each speaker,
    reducing token usage and improving LLM focus on pertinent information.
    
    Args:
        memory: Memory instance containing memories for both speakers
        speaker_1: Name of the first speaker
        speaker_2: Name of the second speaker
        question: Question to answer (used as query for similarity search)
        top_k_per_speaker: Maximum number of relevant memories per speaker (default: 5)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, include all memories (default: True)
        prompt_template_path: Path to qa.txt template (optional, auto-detects if None)
        
    Returns:
        Complete prompt string ready to send to LLM
    """
    # Auto-detect template path if not provided
    if prompt_template_path is None:
        # Look for qa.txt in ../prompts/ relative to this file
        prompt_template_path = Path(__file__).parent.parent / "prompts" / "qa.txt"
    
    # Load the template
    with open(prompt_template_path, "r") as f:
        template = f.read()
    
    # Get memories for each speaker using similarity search
    speaker_1_memories = format_memories_for_speaker(
        memory=memory,
        speaker_name=speaker_1,
        query=question,
        top_k=top_k_per_speaker,
        similarity_threshold=similarity_threshold,
        use_similarity=use_similarity
    )
    speaker_2_memories = format_memories_for_speaker(
        memory=memory,
        speaker_name=speaker_2,
        query=question,
        top_k=top_k_per_speaker,
        similarity_threshold=similarity_threshold,
        use_similarity=use_similarity
    )
    
    # Convert to JSON strings for better formatting
    speaker_1_memories_json = json.dumps(speaker_1_memories, indent=2)
    speaker_2_memories_json = json.dumps(speaker_2_memories, indent=2)
    
    # Replace placeholders in template
    prompt = template.replace("{{speaker_1}}", speaker_1)
    prompt = prompt.replace("{{speaker_2}}", speaker_2)
    prompt = prompt.replace("{{speaker_1_memories}}", speaker_1_memories_json)
    prompt = prompt.replace("{{speaker_2_memories}}", speaker_2_memories_json)
    prompt = prompt.replace("{{question}}", question)
    prompt = prompt.replace("{{session_time}}", session_time)
    
    return prompt