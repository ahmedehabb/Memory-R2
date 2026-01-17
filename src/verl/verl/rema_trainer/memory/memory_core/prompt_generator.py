"""
Prompt generator for memory operations.

This module generates prompts for the LLM to decide what memory operations to perform
based on conversation turns and existing memory state from a Memory instance.
"""

from typing import List, Dict, Any
import json
from pathlib import Path
from verl.rema_trainer.memory.memory_core.memory import Memory


def format_turns_for_prompt(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format conversation turns for the prompt.
    
    Extracts speaker and text, and includes image information if present
    (using BLIP caption).
    
    Args:
        turns: List of turn dicts with keys like session_id, speaker, text, img_url, blip_caption, etc.
        
    Returns:
        List of simplified turn dicts with just speaker and text
        
    Example input:
        [
            {
                "session_id": 9,
                "session_time": "6:59 pm on 26 August, 2023",
                "speaker": "John",
                "img_url": ["https://..."],
                "blip_caption": "a photo of a cityscape with a view of a skyscraper",
                "query": "new york city skyline",
                "text": "Check out this pic from New York City!",
                "dia_id": "D9:6"
            },
            {
                "session_id": 9,
                "session_time": "6:59 pm on 26 August, 2023",
                "speaker": "Tim",
                "text": "Wow! That skyline looks amazing",
                "dia_id": "D9:7"
            }
        ]
        
    Example output:
        [
            {
                "speaker": "John",
                "text": "Check out this pic from New York City! [Sent an image showing a photo of a cityscape with a view of a skyscraper]",
                "dia_id": "D9:6"
            },
            {
                "speaker": "Tim",
                "text": "Wow! That skyline looks amazing",
                "dia_id": "D9:7"
            }
        ]
    """
    formatted_turns = []
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", "")
        
        # Check if there's an image with BLIP caption (sometimes only blip_caption exists without img_url)
        # if turn.get("img_url") and turn.get("blip_caption"):
        if turn.get("blip_caption"):
            # Append image description to the text (text comes first, then image)
            text = f"{text} [Sent an image showing {turn['blip_caption']}]"
        
        formatted_turns.append({
            "speaker": speaker,
            "text": text,
            "dia_id": dia_id
        })
    
    return formatted_turns

def format_memory_for_prompt_for_facts(
    memory: Memory, 
    facts: Dict = None,
    top_k_per_fact: int = 5,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True
) -> List[Dict[str, Any]]:
    """
    Format existing memory from Memory instance for the prompt.
    
    Retrieves relevant memories by searching turn-by-turn, which makes more sense than
    concatenating all turns into one query. For each turn, we search for memories relevant
    to that turn's speaker and content, then deduplicate across all turns.
    
    Args:
        memory: Memory instance
        facts: Dict of facts to use for similarity search (optional)
        top_k_per_fact: Maximum number of relevant memories to retrieve per fact (default: 5)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, return all memories (default: True)
        
    Returns:
        List of simplified memory dicts (deduplicated across all turns)
        
    Example output:
        [
            {
                "memory_id": "a1b2c3d4",
                "speaker": "John",
                "content": "Enjoys outdoor activities",
                "session_time": "5:00 pm",
                "dia_ids": ["D1:3", "D2:5"]
            }
        ]
    """
    formatted_memory = []
    seen_memory_ids = set()  # Track unique memories to avoid duplicates
    facts_list = facts.get("facts", []) if facts else []
    
    # If no facts provided or facts list is empty, return empty memory
    if not facts_list:
        return formatted_memory
    
    if use_similarity:
        
        # Search turn-by-turn for relevant memories
        for fact in facts_list:
            # Extract fact text and speaker
            if isinstance(fact, dict):
                fact_text = fact.get("fact", "")
                fact_speaker = fact.get("speaker", None)
            else:
                continue  # Skip if fact is not a dict
            
            # Ensure fact_text is a string (defensive check)
            if not fact_text or not isinstance(fact_text, str):
                print(f"Warning: Invalid fact_text type: {type(fact_text)}, value: {fact_text}")
                continue
            
            # Search for memories relevant to this specific turn
            # Optionally filter by speaker to get memories ABOUT this speaker
            search_results = memory.search(
                query=fact_text,
                speaker=fact_speaker,  # Get memories about this speaker
                top_k=top_k_per_fact,
                search_method="text-embedding"
            )
            
            # Filter by similarity threshold and format (deduplicate by memory_id)
            for memory_dict, similarity_score in search_results:
                if similarity_score >= similarity_threshold:
                    memory_id = memory_dict.get("memory_id")
                    if memory_id not in seen_memory_ids:
                        seen_memory_ids.add(memory_id)
                        formatted_memory.append({
                            "memory_id": memory_id,
                            "session_time": memory_dict.get("session_time"),
                            "speaker": memory_dict.get("speaker"),
                            "content": memory_dict.get("content"),
                            "dia_ids": memory_dict.get("dia_ids", [])
                        })
    
    return formatted_memory

def format_memory_for_prompt(
    memory: Memory, 
    query_turns: List[Dict[str, Any]] = None,
    top_k: int = 10,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True
) -> List[Dict[str, Any]]:
    """
    Format existing memory from Memory instance for the prompt.
    
    Retrieves relevant memories by searching turn-by-turn, which makes more sense than
    concatenating all turns into one query. For each turn, we search for memories relevant
    to that turn's speaker and content, then deduplicate across all turns.
    
    Args:
        memory: Memory instance
        query_turns: List of conversation turns to use for similarity search (optional)
        top_k: Maximum total number of relevant memories to retrieve across all turns (default: 10)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, return all memories (default: True)
        
    Returns:
        List of simplified memory dicts (deduplicated across all turns)
        
    Example output:
        [
            {
                "memory_id": "a1b2c3d4",
                "speaker": "John",
                "content": "Enjoys outdoor activities"
            }
        ]
    """
    formatted_memory = []
    seen_memory_ids = set()  # Track unique memories to avoid duplicates
    
    if not query_turns:
        return formatted_memory

    if use_similarity and query_turns:
        # Calculate top_k per turn to reach approximately top_k total memories
        # (accounting for potential duplicates across turns)
        num_turns = len(query_turns)
        top_k_per_turn = max(1, top_k // num_turns) if num_turns > 0 else top_k
        
        # Search turn-by-turn for relevant memories
        for turn in query_turns:
            turn_text = turn.get("text", "")
            turn_speaker = turn.get("speaker", None)
            
            # Include image caption in the query if present
            # i only add add blip caption when img_url exists, as found out sometimes it exists alone and doesnt make sense !!
            # TODO:: SHOULD WE SEARCH FOR MEMORIES ABOUT THE IMAGE TOO?? LIKE IF THE IMAGE IS OF A BEACH, SEARCH FOR MEMORIES ABOUT BEACHES??
            if turn.get("img_url") and turn.get("blip_caption"):
                turn_text = f"{turn_text} {turn['blip_caption']}"
            
            if not turn_text:
                continue
            
            # Search for memories relevant to this specific turn
            # Optionally filter by speaker to get memories ABOUT this speaker
            search_results = memory.search(
                query=turn_text,
                speaker=turn_speaker,  # Get memories about this speaker
                top_k=top_k_per_turn,
                search_method="text-embedding"
            )
            
            # Filter by similarity threshold and format (deduplicate by memory_id)
            for memory_dict, similarity_score in search_results:
                if similarity_score >= similarity_threshold:
                    memory_id = memory_dict.get("memory_id")
                    if memory_id not in seen_memory_ids:
                        seen_memory_ids.add(memory_id)
                        formatted_memory.append({
                            "memory_id": memory_id,
                            "speaker": memory_dict.get("speaker"),
                            "content": memory_dict.get("content")
                        })
    else:
        # Fallback: return all memories (original behavior)
        memory_list = memory.get()  # Get all memories from Memory instance
        
        for mem in memory_list:
            formatted_memory.append({
                "memory_id": mem.get("memory_id"),
                "speaker": mem.get("speaker"),
                "content": mem.get("content")
            })
    
    return formatted_memory

def generate_memory_prompt_using_facts(
    memory: Memory,  # Memory instance
    facts: Dict, 
    prompt_template_path: str = None,
    top_k_memories: int = 40,
    similarity_threshold: float = 0.1,
    use_similarity: bool = True
) -> str:
    """
    Generate the complete prompt for memory operations.
    
    Uses turn-by-turn similarity search to retrieve relevant memories. Each turn is used
    as a separate query to find memories about that turn's speaker and content, then results
    are deduplicated. This is more focused than concatenating all turns into one query.
    
    Args:
        memory: Memory instance with existing memories
        turns: List of conversation turn dicts
        prompt_template_path: Path to memory.txt template (optional, auto-detects if None)
        top_k_memories: Maximum total number of relevant memories to retrieve (default: 20)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, include all memories (default: True)
        
    Returns:
        Complete prompt string ready to send to LLM
        formatted_turns: List of formatted turns for debugging or further processing
        formatted_memory: List of formatted memories for debugging or further processing
    """
    # Auto-detect template path if not provided
    if prompt_template_path is None:
        # Look for memory_v2.txt in ../prompts/ relative to this file
        prompt_template_path = Path(__file__).parent.parent / "prompts" / "memory_v2.txt"
    
    # Load the template
    with open(prompt_template_path, "r") as f:
        template = f.read()
    
    # Format facts and memory (with turn-by-turn similarity search)
    formatted_memory = format_memory_for_prompt_for_facts(
        memory, 
        facts=facts,
        top_k=top_k_memories,
        similarity_threshold=similarity_threshold,
        use_similarity=use_similarity
    )
    
    # Convert to JSON strings
    facts_json = json.dumps(facts, indent=2)
    memory_json = json.dumps(formatted_memory, indent=2)
    
    # Replace placeholders in template
    prompt = template.replace("{existing_memory}", memory_json)
    prompt = prompt.replace("{new_facts}", facts_json)
    
    return prompt


def generate_memory_prompt(
    memory: Memory,  # Memory instance
    turns: List[Dict[str, Any]], 
    prompt_template_path: str = None,
    top_k_memories: int = 20,
    similarity_threshold: float = 0.3,
    use_similarity: bool = True
) -> str:
    """
    Generate the complete prompt for memory operations.
    
    Uses turn-by-turn similarity search to retrieve relevant memories. Each turn is used
    as a separate query to find memories about that turn's speaker and content, then results
    are deduplicated. This is more focused than concatenating all turns into one query.
    
    Args:
        memory: Memory instance with existing memories
        turns: List of conversation turn dicts
        prompt_template_path: Path to memory.txt template (optional, auto-detects if None)
        top_k_memories: Maximum total number of relevant memories to retrieve (default: 20)
        similarity_threshold: Minimum similarity score to include a memory (default: 0.3)
        use_similarity: If True, use similarity search; if False, include all memories (default: True)
        
    Returns:
        Complete prompt string ready to send to LLM
        formatted_turns: List of formatted turns for debugging or further processing
        formatted_memory: List of formatted memories for debugging or further processing
    """
    # Auto-detect template path if not provided
    if prompt_template_path is None:
        # Look for memory_v2.txt in ../prompts/ relative to this file
        prompt_template_path = Path(__file__).parent.parent / "prompts" / "memory_v2.txt"
    
    # Load the template
    with open(prompt_template_path, "r") as f:
        template = f.read()
    
    # Format turns and memory (with turn-by-turn similarity search)
    formatted_turns = format_turns_for_prompt(turns)
    formatted_memory = format_memory_for_prompt(
        memory, 
        query_turns=turns,
        top_k=top_k_memories,
        similarity_threshold=similarity_threshold,
        use_similarity=use_similarity
    )
    
    # Convert to JSON strings
    turns_json = json.dumps(formatted_turns, indent=2)
    memory_json = json.dumps(formatted_memory, indent=2)
    
    # Replace placeholders in template
    prompt = template.replace("{existing_memory}", memory_json)
    prompt = prompt.replace("{new_turns}", turns_json)
    
    return prompt, formatted_turns, formatted_memory


def generate_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    prompt_template_path: str = None
) -> str:
    """
    Generate the LLM-as-a-judge prompt for answer evaluation.
    
    Args:
        question: The question that was asked
        gold_answer: The reference/gold answer
        predicted_answer: The model's predicted answer
        prompt_template_path: Path to llm_as_a_judge.txt template (optional, auto-detects if None)
        
    Returns:
        Complete judge prompt string ready to send to LLM
    """
    # Auto-detect template path if not provided
    if prompt_template_path is None:
        # Look for llm_as_a_judge.txt in ../prompts/ relative to this file
        prompt_template_path = Path(__file__).parent.parent / "prompts" / "llm_as_a_judge.txt"
    
    # Load the template
    with open(prompt_template_path, "r") as f:
        template = f.read()
    
    # Replace placeholders in template
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{gold_answer}}", gold_answer)
    prompt = prompt.replace("{{predicted_answer}}", predicted_answer)
    
    return prompt

def generate_memory_judge_prompt(
    turns: List[Dict[str, Any]],
    memory_state: List[Dict[str, Any]],
    memory_operations: List[Dict[str, Any]],
    prompt_template_path: str = None
) -> str:
    """
    Generate the LLM-as-a-judge prompt for answer evaluation.
    
    Args:
        turns: List of conversation turn dicts
        memory_state: List of existing memory dicts
        memory_operations: List of memory operation dicts performed
        prompt_template_path: Path to llm_as_a_judge.txt template (optional, auto-detects if None)

    Returns:
        Complete judge prompt string ready to send to LLM
    """
    # Auto-detect template path if not provided
    if prompt_template_path is None:
        # Look for llm_as_a_judge.txt in ../prompts/ relative to this file
        prompt_template_path = Path(__file__).parent.parent / "prompts" / "eval_memory.txt"
    
    # Load the template
    with open(prompt_template_path, "r") as f:
        template = f.read()
    
    # Replace placeholders in template
    prompt = template.replace("{{existing_memory}}", json.dumps(memory_state, indent=2))
    prompt = prompt.replace("{{new_turns}}", json.dumps(turns, indent=2))
    prompt = prompt.replace("{{model_operations}}", json.dumps(memory_operations, indent=2))
    
    return prompt


# Example usage
if __name__ == "__main__":
    
    # Example: Initialize memory with some state
    memory = Memory(embedding_method="openai")
    memory.insert("conv-41", 5, "5:00 pm", "Tim", "Enjoys reading books from various genres")
    memory.insert("conv-41", 7, "5:30 pm", "John", "Likes exploring new places and traveling")
    
    # Example turns (as provided)
    turns = [
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "John",
            "img_url": ["https://i.pinimg.com/originals/90/49/55/904955fe77567cf689d7db0ce606717d.jpg"],
            "blip_caption": "a photo of a cityscape with a view of a skyscraper",
            "query": "new york city skyline",
            "dia_id": "D9:6",
            "text": "Wow, Tim, that's an awesome book collection! It's cool to escape to different worlds with a hobby. By the way, I love discovering new cities - check out this pic from one of my trips to New York City!"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "Tim",
            "dia_id": "D9:7",
            "text": "Wow! That skyline looks amazing - I've been wanting to visit NYC. How was it?"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "John",
            "dia_id": "D9:8",
            "text": "Thanks! It was amazing. Everywhere you go there's something new and exciting. Exploring the city and trying all the restaurants was awesome. It's a must-visit!"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "Tim",
            "dia_id": "D9:9",
            "text": "Adding NYC to my travel list, sounds like a great adventure! I heard there's so much to explore and try out. Can't wait to visit!"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "John",
            "dia_id": "D9:10",
            "text": "Trust me, NYC is amazing! It's got so much to check out - the culture, food - you won't regret it. It's an adventure you'll never forget!"
        }
    ]
    
    # Generate prompt
    prompt, formatted_turns, formatted_memory = generate_memory_prompt(memory, turns)
    
    # Print formatted turns to show the transformation
    print("="*70)
    print("FORMATTED TURNS:")
    print("="*70)
    formatted = format_turns_for_prompt(turns)
    print(json.dumps({"turns": formatted}, indent=2))
    
    print("\n" + "="*70)
    print("FORMATTED MEMORY:")
    print("="*70)
    formatted_mem = format_memory_for_prompt(memory)
    print(json.dumps(formatted_mem, indent=2))
    
    print("\n" + "="*70)
    print("COMPLETE PROMPT (last 500 chars):")
    print("="*70)
    print(prompt[-500:])
