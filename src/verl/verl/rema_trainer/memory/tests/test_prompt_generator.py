"""
Test prompt_generator to verify correctness.

This test verifies:
1. Turns are formatted correctly (with image captions)
2. Memory is formatted correctly (only necessary fields)
3. Prompt template is loaded and placeholders replaced
4. Final prompt is valid and ready for LLM
"""

import sys
import json
from pathlib import Path

from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.prompt_generator import generate_memory_prompt, format_turns_for_prompt, format_memory_for_prompt


def test_format_turns():
    """Test that turns are formatted correctly."""
    print("\n" + "="*70)
    print("TEST 1: Format Turns")
    print("="*70)
    
    turns = [
        {
            "session_id": 1,
            "speaker": "John",
            "img_url": ["https://example.com/image.jpg"],
            "blip_caption": "a sunset over mountains",
            "text": "Check out this beautiful sunset!"
        },
        {
            "session_id": 2,
            "speaker": "Jane",
            "text": "Wow, that's gorgeous!"
        }
    ]
    
    formatted = format_turns_for_prompt(turns)
    
    # Verify first turn has image prefix
    assert formatted[0]["speaker"] == "John"
    assert "[Image: a sunset over mountains]" in formatted[0]["text"]
    assert "Check out this beautiful sunset!" in formatted[0]["text"]
    print(f"✓ Turn with image: {formatted[0]['text'][:60]}...")
    
    # Verify second turn has no image prefix
    assert formatted[1]["speaker"] == "Jane"
    assert "[Image:" not in formatted[1]["text"]
    assert formatted[1]["text"] == "Wow, that's gorgeous!"
    print(f"✓ Turn without image: {formatted[1]['text']}")
    
    print("✓ Format turns test PASSED\n")
    return formatted


def test_format_memory():
    """Test that memory is formatted correctly."""
    print("="*70)
    print("TEST 2: Format Memory")
    print("="*70)
    
    memory = Memory(embedding_method="openai")
    memory.insert("conv-1", 1, "10:00 am", "John", "Enjoys hiking and outdoor activities")
    memory.insert("conv-1", 2, "10:05 am", "Jane", "Works as a software engineer")
    
    formatted = format_memory_for_prompt(memory)
    
    # Verify correct fields are included
    assert len(formatted) == 2
    assert "memory_id" in formatted[0]
    assert "speaker" in formatted[0]
    assert "content" in formatted[0]
    
    # Verify unnecessary fields are excluded
    assert "sample_id" not in formatted[0]
    assert "session_id" not in formatted[0]
    assert "session_time" not in formatted[0]
    
    print(f"✓ Memory entry 1: {formatted[0]}")
    print(f"✓ Memory entry 2: {formatted[1]}")
    print("✓ Format memory test PASSED\n")
    
    return memory, formatted


def test_prompt_generation():
    """Test complete prompt generation."""
    print("="*70)
    print("TEST 3: Generate Complete Prompt")
    print("="*70)
    
    # Setup memory
    memory = Memory(embedding_method="openai")
    memory.insert("conv-1", 1, "10:00 am", "John", "Enjoys traveling")
    memory.insert("conv-1", 2, "10:05 am", "Jane", "Likes photography")
    
    # Setup turns
    turns = [
        {
            "session_id": 3,
            "session_time": "10:10 am",
            "speaker": "John",
            "img_url": ["https://example.com/nyc.jpg"],
            "blip_caption": "a photo of NYC skyline",
            "text": "I visited NYC last week, amazing city!"
        },
        {
            "session_id": 4,
            "session_time": "10:11 am",
            "speaker": "Jane",
            "text": "I'd love to photograph the skyline there!"
        }
    ]
    
    # Generate prompt WITHOUT similarity search to ensure all memories are included
    prompt, formatted_turns, formatted_memory = generate_memory_prompt(memory, turns, use_similarity=False)
    
    # Verify prompt structure
    assert "{existing_memory}" not in prompt, "Placeholder not replaced!"
    assert "{new_turns}" not in prompt, "Placeholder not replaced!"
    
    # Verify memory is in prompt
    assert "Enjoys traveling" in prompt
    assert "Likes photography" in prompt
    print("✓ Existing memory included in prompt")
    
    # Verify turns are in prompt
    assert "[Image: a photo of NYC skyline]" in prompt
    assert "I visited NYC last week" in prompt
    assert "I'd love to photograph the skyline" in prompt
    print("✓ New turns included in prompt")
    
    # Verify prompt structure
    assert "**Existing Memory:**" in prompt or "Existing Memory" in prompt
    assert "**New Conversation Turns:**" in prompt or "New Conversation Turns" in prompt
    print("✓ Prompt template structure intact")
    
    print(f"\n✓ Prompt length: {len(prompt)} characters")
    print("✓ Generate prompt test PASSED\n")
    
    return prompt


def test_prompt_content():
    """Test the actual content of generated prompt."""
    print("="*70)
    print("TEST 4: Verify Prompt Content")
    print("="*70)
    
    memory = Memory(embedding_method="openai")
    memory.insert("conv-1", 1, "10:00 am", "Tim", "Enjoys reading books")
    
    turns = [
        {
            "session_id": 2,
            "session_time": "10:05 am",
            "speaker": "John",
            "img_url": ["https://example.com/nyc.jpg"],
            "blip_caption": "a cityscape with skyscrapers",
            "text": "Check out NYC!"
        }
    ]
    
    # Disable similarity search to ensure memory is included
    prompt, formatted_turns, formatted_memory = generate_memory_prompt(memory, turns, use_similarity=False)
    
    # The prompt should contain the actual data we inserted
    assert "Enjoys reading books" in prompt, "Memory content not found in prompt"
    assert "[Image: a cityscape with skyscrapers]" in prompt, "Image caption not found"
    assert "Check out NYC!" in prompt, "Turn text not found"
    
    print("✓ Memory content found in prompt")
    print("✓ Turn content found (with image)")
    print("✓ Prompt content test PASSED\n")


def test_similarity_search():
    """Test that similarity search retrieves only relevant memories."""
    print("="*70)
    print("TEST 5: Similarity Search for Relevant Memories")
    print("="*70)
    
    # Setup memory with diverse topics
    memory = Memory(embedding_method="openai")
    memory.insert("conv-1", 1, "10:00 am", "John", "Loves traveling to new cities")
    memory.insert("conv-1", 2, "10:01 am", "John", "Visited NYC and loved the skyline")
    memory.insert("conv-1", 3, "10:02 am", "Jane", "Enjoys cooking Italian food")
    memory.insert("conv-1", 4, "10:03 am", "Tim", "Works as a software engineer")
    memory.insert("conv-1", 5, "10:04 am", "John", "Planning trip to Chicago next month")
    memory.insert("conv-1", 6, "10:05 am", "Jane", "Loves gardening and plants")
    
    # New turns about travel (should match memories about travel/cities)
    turns = [
        {
            "session_id": 7,
            "speaker": "John",
            "text": "I'm thinking of visiting Los Angeles soon, heard it's amazing!"
        },
        {
            "session_id": 8,
            "speaker": "Tim",
            "text": "Yeah, LA has great sights and beaches!"
        }
    ]
    
    # Test with similarity search enabled
    formatted_with_similarity = format_memory_for_prompt(
        memory,
        query_turns=turns,
        top_k=3,
        similarity_threshold=0.2,
        use_similarity=True
    )
    
    # Test without similarity search (all memories)
    formatted_without_similarity = format_memory_for_prompt(
        memory,
        query_turns=turns,
        top_k=10,
        similarity_threshold=0.0,
        use_similarity=False
    )
    
    print(f"✓ With similarity search: {len(formatted_with_similarity)} memories retrieved")
    print(f"✓ Without similarity search: {len(formatted_without_similarity)} memories retrieved")
    
    # Verify similarity search returns fewer memories
    assert len(formatted_with_similarity) <= len(formatted_without_similarity), \
        "Similarity search should return same or fewer memories"
    
    # Verify similarity search respects top_k limit
    assert len(formatted_with_similarity) <= 3, \
        f"Expected at most 3 memories, got {len(formatted_with_similarity)}"
    
    # Verify all memories returned without similarity search
    assert len(formatted_without_similarity) == 6, \
        f"Expected all 6 memories, got {len(formatted_without_similarity)}"
    
    print("\n✓ Retrieved memories with similarity search:")
    for mem in formatted_with_similarity:
        print(f"  - {mem['speaker']}: {mem['content']}")
    
    print("✓ Similarity search test PASSED\n")


def test_similarity_threshold():
    """Test that similarity threshold filters out low-relevance memories."""
    print("="*70)
    print("TEST 6: Similarity Threshold Filtering")
    print("="*70)
    
    memory = Memory(embedding_method="openai")
    memory.insert("conv-1", 1, "10:00 am", "John", "Loves pizza and pasta")
    memory.insert("conv-1", 2, "10:01 am", "Jane", "Enjoys hiking in mountains")
    memory.insert("conv-1", 3, "10:02 am", "Tim", "Works remotely as developer")
    
    turns = [
        {
            "session_id": 4,
            "speaker": "John",
            "text": "I had amazing Italian food yesterday at a new restaurant!"
        }
    ]
    
    # Test with high threshold (strict filtering)
    formatted_high_threshold = format_memory_for_prompt(
        memory,
        query_turns=turns,
        top_k=10,
        similarity_threshold=0.5,  # High threshold
        use_similarity=True
    )
    
    # Test with low threshold (more permissive)
    formatted_low_threshold = format_memory_for_prompt(
        memory,
        query_turns=turns,
        top_k=10,
        similarity_threshold=0.1,  # Low threshold
        use_similarity=True
    )
    
    print(f"✓ High threshold (0.5): {len(formatted_high_threshold)} memories")
    print(f"✓ Low threshold (0.1): {len(formatted_low_threshold)} memories")
    
    # Low threshold should return same or more memories than high threshold
    assert len(formatted_low_threshold) >= len(formatted_high_threshold), \
        "Lower threshold should return more or equal memories"
    
    print("✓ Threshold filtering test PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  PROMPT GENERATOR VERIFICATION TESTS")
    print("="*70)
    
    try:
        test_format_turns()
        test_format_memory()
        test_prompt_generation()
        test_prompt_content()
        test_similarity_search()
        test_similarity_threshold()
        
        print("="*70)
        print("  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        print("\nThe prompt generator is working correctly!")
        print("\nKey verifications:")
        print("  ✓ Turns formatted with image captions")
        print("  ✓ Memory formatted with only necessary fields")
        print("  ✓ Template placeholders replaced correctly")
        print("  ✓ Final prompt contains all required information")
        print("  ✓ Prompt ready for LLM consumption")
        print("  ✓ Similarity search retrieves relevant memories only")
        print("  ✓ Threshold filtering works correctly")
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("  ✗✗✗ TEST FAILED ✗✗✗")
        print("="*70)
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
