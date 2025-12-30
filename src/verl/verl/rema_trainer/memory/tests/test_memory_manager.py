"""
Test suite for MemoryManager.
Tests JSON command parsing, execution, and error handling for conversation-based memory.
"""
import json
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager, create_function_schema


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(result: dict):
    """Pretty print a result dict."""
    print(f"Status: {result['status']}")
    print(f"Message: {result.get('message', 'N/A')}")
    if result.get('data'):
        print(f"Data: {json.dumps(result['data'], indent=2)}")
    print()


def test_insert_operations():
    """Test insert conversation turn operations via JSON commands."""
    print_section("Test 1: Insert Operations")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Test 1: Insert conversation turn (dict)
    print("1.1 Insert conversation turn (dict):")
    command = {
        "operation": "insert",
        "sample_id": "conv-41",
        "session_id": 1,
        "session_time": "11:01 am on 17 December, 2022",
        "speaker": "John",
        "content": "I'm interested in learning Python programming"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Insert should succeed"
    assert "memory_id" in result, "Should return memory_id"
    
    # Test 2: Insert conversation turn (JSON string)
    print("1.2 Insert conversation turn (JSON string):")
    command_json = json.dumps({
        "operation": "insert",
        "sample_id": "conv-41",
        "session_id": 2,
        "session_time": "11:05 am on 17 December, 2022",
        "speaker": "AI",
        "content": "Python is great for beginners! It has clear syntax and many learning resources."
    })
    result = manager.execute_command(mem, command_json)
    print_result(result)
    assert result["status"] == "success", "Insert should succeed"
    
    # Test 3: Insert from different conversation
    print("1.3 Insert from different conversation:")
    command = {
        "operation": "insert",
        "sample_id": "conv-42",
        "session_id": 1,
        "session_time": "2:30 pm on 18 December, 2022",
        "speaker": "Alice",
        "content": "What's the best way to learn machine learning?"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Insert should succeed"
    
    # Test 5: Missing sample_id
    print("1.4 Insert without sample_id:")
    command = {
        "operation": "insert",
        "session_id": 1,
        "session_time": "11:01 am",
        "speaker": "John",
        "content": "Some content"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 6: Missing session_id
    print("1.5 Insert without session_id:")
    command = {
        "operation": "insert",
        "sample_id": "conv-41",
        "session_time": "11:01 am",
        "speaker": "John",
        "content": "Some content"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 7: Missing content
    print("1.6 Insert without content:")
    command = {
        "operation": "insert",
        "sample_id": "conv-41",
        "session_id": 1,
        "session_time": "11:01 am",
        "speaker": "John"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    print(f"✓ Memory stats: {manager.get_memory_stats(mem)}")


def test_update_operations():
    """Test update conversation turn operations."""
    print_section("Test 2: Update Operations")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Insert a turn first
    insert_result = manager.execute_command(mem, {
        "operation": "insert",
        "sample_id": "conv-update-test",
        "session_id": 1,
        "session_time": "3:00 pm",
        "speaker": "User",
        "content": "Original content to be updated"
    })
    memory_id = insert_result["memory_id"]
    print(f"Inserted turn with ID: {memory_id}")
    print(f"Original content: '{insert_result['data']['content']}'")
    
    # Test 1: Update the turn
    print("\n2.1 Update conversation turn:")
    command = {
        "operation": "update",
        "memory_id": memory_id,
        "content": "This is the new updated content!"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Update should succeed"
    assert result["data"]["content"] == "This is the new updated content!"
    print(f"Updated content: '{result['data']['content']}'")
    
    # Test 2: Update with missing memory_id
    print("2.2 Update without memory_id:")
    command = {
        "operation": "update",
        "content": "New content"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 3: Update with missing content
    print("2.3 Update without content:")
    command = {
        "operation": "update",
        "memory_id": memory_id
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 5: Update non-existent turn
    print("2.4 Update non-existent turn:")
    command = {
        "operation": "update",
        "memory_id": "nonexistent123",
        "content": "New content"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"


def test_delete_operations():
    """Test delete conversation turn operations."""
    print_section("Test 3: Delete Operations")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Insert a turn first
    insert_result = manager.execute_command(mem, {
        "operation": "insert",
        "sample_id": "conv-50",
        "session_id": 1,
        "session_time": "3:00 pm",
        "speaker": "Bob",
        "content": "Temporary conversation turn to delete"
    })
    memory_id = insert_result["memory_id"]
    print(f"Inserted turn with ID: {memory_id}")
    
    stats_before = manager.get_memory_stats(mem)
    print(f"Total turns before delete: {stats_before['total_turns']}")
    
    # Test 1: Delete the turn
    print("\n3.1 Delete conversation turn:")
    command = {
        "operation": "delete",
        "memory_id": memory_id
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Delete should succeed"
    
    stats_after = manager.get_memory_stats(mem)
    print(f"Total turns after delete: {stats_after['total_turns']}")
    assert stats_after['total_turns'] == stats_before['total_turns'] - 1
    
    # Test 2: Delete with missing memory_id
    print("3.2 Delete without memory_id:")
    command = {
        "operation": "delete"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 3: Delete non-existent turn
    print("3.3 Delete non-existent turn:")
    command = {
        "operation": "delete",
        "memory_id": "nonexistent123"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"


def test_search_operations():
    """Test search conversation turn operations."""
    print_section("Test 4: Search Operations")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Insert some test conversation turns
    print("Inserting test conversation turns...")
    test_turns = [
        ("conv-41", 1, "John", "I want to learn Python programming"),
        ("conv-41", 2, "AI", "Python is great for beginners and has many libraries"),
        ("conv-41", 3, "John", "What about machine learning with Python?"),
        ("conv-41", 4, "AI", "Python has excellent ML libraries like scikit-learn and TensorFlow"),
        ("conv-42", 1, "Alice", "I'm interested in deep learning and neural networks"),
        ("conv-42", 2, "AI", "Deep learning uses neural networks inspired by the brain"),
    ]
    
    for sample_id, session_id, speaker, content in test_turns:
        manager.execute_command(mem, {
            "operation": "insert",
            "sample_id": sample_id,
            "session_id": session_id,
            "session_time": f"{session_id}:00 pm",
            "speaker": speaker,
            "content": content
        })
    
    print(f"Inserted {len(test_turns)} conversation turns\n")
    
    # Test 1: BM25 search
    print("4.1 BM25 search for 'machine learning':")
    command = {
        "operation": "search",
        "query": "machine learning",
        "top_k": 3,
        "search_method": "bm25"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    assert len(result["data"]) <= 3, "Should return at most top_k results"
    
    # Test 2: Embedding search
    print("4.2 Embedding search for 'neural networks':")
    command = {
        "operation": "search",
        "query": "neural networks",
        "top_k": 2,
        "search_method": "text-embedding"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    
    # Test 3: Search with sample_id filter
    print("4.3 Search within specific conversation (conv-41):")
    command = {
        "operation": "search",
        "query": "Python",
        "sample_id": "conv-41",
        "top_k": 10,
        "search_method": "bm25"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    # All results should be from conv-41
    for turn in result["data"]:
        assert turn["sample_id"] == "conv-41", "All results should be from conv-41"
    
    # Test 5: Search with speaker filter
    print("4.4 Search by speaker (John):")
    command = {
        "operation": "search",
        "query": "learning",
        "speaker": "John",
        "top_k": 10,
        "search_method": "bm25"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    # All results should be from John
    for turn in result["data"]:
        assert turn["speaker"] == "John", "All results should be from John"
    
    # Test 6: Search with both filters
    print("4.5 Search with sample_id AND speaker filters:")
    command = {
        "operation": "search",
        "query": "Python",
        "sample_id": "conv-41",
        "speaker": "AI",
        "search_method": "bm25"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    # All results should match both filters
    for turn in result["data"]:
        assert turn["sample_id"] == "conv-41" and turn["speaker"] == "AI"
    
    # Test 7: Search with min_score
    print("4.6 Search with min_score threshold:")
    command = {
        "operation": "search",
        "query": "neural networks",
        "min_score": 0.3,
        "search_method": "text-embedding"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "success", "Search should succeed"
    
    # Test 8: Invalid search method
    print("4.7 Search with invalid method:")
    command = {
        "operation": "search",
        "query": "test",
        "search_method": "invalid_method"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"
    
    # Test 9: Missing query
    print("4.8 Search without query:")
    command = {
        "operation": "search",
        "search_method": "bm25"
    }
    result = manager.execute_command(mem, command)
    print_result(result)
    assert result["status"] == "error", "Should return error"


def test_batch_operations():
    """Test batch command execution."""
    print_section("Test 5: Batch Operations")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Test 1: Batch insert and search
    print("5.1 Batch insert and search:")
    batch_commands = [
        {
            "operation": "insert",
            "sample_id": "conv-100",
            "session_id": 1,
            "session_time": "10:00 am",
            "speaker": "User",
            "content": "Tell me about JavaScript"
        },
        {
            "operation": "insert",
            "sample_id": "conv-100",
            "session_id": 2,
            "session_time": "10:01 am",
            "speaker": "AI",
            "content": "JavaScript is a scripting language used for web development"
        },
        {
            "operation": "insert",
            "sample_id": "conv-100",
            "session_id": 3,
            "session_time": "10:02 am",
            "speaker": "User",
            "content": "What about TypeScript?"
        },
        {
            "operation": "search",
            "query": "JavaScript programming",
            "sample_id": "conv-100",
            "top_k": 5,
            "search_method": "bm25"
        }
    ]
    
    result = manager.execute_batch(mem, batch_commands)
    print(f"Status: {result['status']}")
    print(f"Total: {result['total_commands']}, Success: {result['successful']}, Failed: {result['failed']}\n")
    
    for item in result['results']:
        print(f"Command {item['command_index']}: {item['result']['status']}")
        if item['result']['status'] == 'success' and item['command']['operation'] == 'search':
            print(f"  Found {len(item['result']['data'])} results")
    
    assert result['successful'] == len(batch_commands), "All commands should succeed"
    
    # Test 2: Batch with JSON string
    print("\n5.2 Batch from JSON string:")
    batch_json = json.dumps([
        {
            "operation": "insert",
            "sample_id": "conv-101",
            "session_id": 1,
            "session_time": "11:00 am",
            "speaker": "User",
            "content": "What is Rust?"
        },
        {
            "operation": "insert",
            "sample_id": "conv-101",
            "session_id": 2,
            "session_time": "11:01 am",
            "speaker": "AI",
            "content": "Rust is a systems programming language"
        }
    ])
    result = manager.execute_batch(mem, batch_json)
    print(f"Executed {result['total_commands']} commands, {result['successful']} successful")
    
    # Test 3: Batch with some errors
    print("\n5.3 Batch with mixed success/failure:")
    batch_commands = [
        {
            "operation": "insert",
            "sample_id": "conv-102",
            "session_id": 1,
            "session_time": "12:00 pm",
            "speaker": "User",
            "content": "Valid turn"
        },
        {
            "operation": "insert",
            "sample_id": "conv-102",
            # Missing session_id - should fail
            "session_time": "12:01 pm",
            "speaker": "AI",
            "content": "Invalid turn"
        },
        {
            "operation": "insert",
            "sample_id": "conv-102",
            "session_id": 3,
            "session_time": "12:02 pm",
            "speaker": "User",
            "content": "Another valid turn"
        }
    ]
    result = manager.execute_batch(mem, batch_commands)
    print(f"Total: {result['total_commands']}, Success: {result['successful']}, Failed: {result['failed']}")
    assert result['status'] == 'partial', "Should be partial success"


def test_operation_history():
    """Test operation history tracking."""
    print_section("Test 6: Operation History")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Perform some operations
    manager.execute_command(mem, {
        "operation": "insert",
        "sample_id": "conv-200",
        "session_id": 1,
        "session_time": "1:00 pm",
        "speaker": "User",
        "content": "History test 1"
    })
    manager.execute_command(mem, {
        "operation": "insert",
        "sample_id": "conv-200",
        "session_id": 2,
        "session_time": "1:01 pm",
        "speaker": "AI",
        "content": "History test 2"
    })
    manager.execute_command(mem, {
        "operation": "search",
        "query": "test",
        "search_method": "bm25"
    })
    
    # Get history
    history = manager.get_operation_history()
    print(f"Total operations in history: {len(history)}")
    assert len(history) == 3, "Should have 3 operations"
    
    # Get limited history
    recent = manager.get_operation_history(limit=2)
    print(f"Recent operations (limit=2): {len(recent)}")
    assert len(recent) == 2, "Should have 2 operations"
    
    # Clear history
    manager.clear_history()
    history_after = manager.get_operation_history()
    print(f"Operations after clear: {len(history_after)}")
    assert len(history_after) == 0, "History should be empty"
    print("✓ History tracking works correctly")


def test_function_schema():
    """Test OpenAI function schema generation."""
    print_section("Test 7: Function Schema Generation")
    
    schema = create_function_schema()
    print(f"Generated {len(schema)} function schemas:")
    
    for func in schema:
        print(f"\n  - {func['name']}: {func['description']}")
        required_params = func['parameters'].get('required', [])
        print(f"    Required params: {', '.join(required_params)}")
    
    assert len(schema) == 4, "Should have 4 function schemas"
    expected_names = ["insert_memory", "search_memory", "update_memory", "delete_memory"]
    actual_names = [f['name'] for f in schema]
    assert set(actual_names) == set(expected_names), f"Expected {expected_names}, got {actual_names}"
    print("\n✓ Function schemas generated successfully")


def test_error_handling():
    """Test various error scenarios."""
    print_section("Test 8: Error Handling")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Test 1: Invalid JSON
    print("8.1 Invalid JSON string:")
    result = manager.execute_command(mem, "not a valid json {")
    print_result(result)
    assert result["status"] == "error"
    
    # Test 2: Missing operation field
    print("8.2 Missing operation field:")
    result = manager.execute_command(mem, {"sample_id": "conv-41", "content": "test"})
    print_result(result)
    assert result["status"] == "error"
    
    # Test 3: Unknown operation
    print("8.3 Unknown operation:")
    result = manager.execute_command(mem, {"operation": "unknown_op", "data": "test"})
    print_result(result)
    assert result["status"] == "error"
    
    # Test 5: Non-dict command
    print("8.4 Non-dict command:")
    result = manager.execute_command(mem, ["not", "a", "dict"])
    print_result(result)
    assert result["status"] == "error"
    
    print("✓ Error handling works correctly")


def test_memory_stats():
    """Test memory statistics."""
    print_section("Test 9: Memory Statistics")
    
    mem = Memory(embedding_method="openai")
    manager = MemoryManager(embedding_method="openai")
    
    # Add some conversation turns
    for i in range(5):
        manager.execute_command(mem, {
            "operation": "insert",
            "sample_id": "conv-300",
            "session_id": i + 1,
            "session_time": f"{i+1}:00 pm",
            "speaker": "User" if i % 2 == 0 else "AI",
            "content": f"Turn {i}"
        })
    
    for i in range(3):
        manager.execute_command(mem, {
            "operation": "insert",
            "sample_id": "conv-301",
            "session_id": i + 1,
            "session_time": f"{i+1}:00 pm",
            "speaker": "Alice",
            "content": f"Event {i}"
        })
    
    # Get stats
    stats = manager.get_memory_stats(mem)
    print("Memory Statistics:")
    print(json.dumps(stats, indent=2))
    
    assert stats['total_turns'] == 8, "Should have 8 total turns"
    assert stats['unique_conversations'] == 2, "Should have 2 unique conversations"
    assert stats['unique_speakers'] == 3, "Should have 3 unique speakers (User, AI, Alice)"
    assert stats['total_operations'] > 0
    print("\n✓ Statistics tracking works correctly")


def test_attach_turn_metadata():
    """Test attaching metadata from turns to operations."""
    print_section("Test 10: Attach Turn Metadata to Operations")
    
    # Create manager instance for testing
    manager = MemoryManager(embedding_method="openai")
    
    # Sample conversation turns - ALL FROM THE SAME SESSION
    # (Important: all turns must have same session_id and session_time)
    # Note: turns don't contain sample_id, it's passed separately as conv_id
    conv_id = "conv-41"
    turns = [
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "John",
            "text": "I went to New York City last week"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "Maria",
            "text": "That sounds exciting! What did you see?"
        },
        {
            "session_id": 9,
            "session_time": "6:59 pm on 26 August, 2023",
            "speaker": "John",
            "text": "I visited the Statue of Liberty and Times Square"
        }
    ]
    
    # Test 1: Operations without metadata (typical LLM output)
    print("10.1 Attach metadata to insert operations:")
    operations = [
        {
            "operation": "insert",
            "speaker": "John",
            "content": "Recently visited New York City, saw Statue of Liberty and Times Square"
        },
        {
            "operation": "insert",
            "speaker": "Maria",
            "content": "Interested in John's travel experiences"
        }
    ]
    
    print("Before:")
    print(json.dumps(operations, indent=2))
    
    enriched_ops = manager.attach_turn_metadata_to_operations(operations, turns, conv_id)
    
    print("\nAfter:")
    print(json.dumps(enriched_ops, indent=2))
    
    # Verify metadata was attached (all turns have same session metadata)
    assert enriched_ops[0]["sample_id"] == "conv-41"
    assert enriched_ops[0]["session_id"] == 9  # Same for all turns in session
    assert enriched_ops[0]["session_time"] == "6:59 pm on 26 August, 2023"
    assert enriched_ops[1]["sample_id"] == "conv-41"
    assert enriched_ops[1]["session_id"] == 9  # Same for all turns in session
    print("✓ Metadata attached correctly (all turns from same session)")
    
    # Test 2: Operations that already have some metadata (shouldn't overwrite)
    print("\n10.2 Don't overwrite existing metadata:")
    operations = [
        {
            "operation": "insert",
            "speaker": "John",
            "content": "Summary text",
            "sample_id": "conv-99",  # Already has sample_id
            "session_id": 999  # Already has session_id
        }
    ]
    
    enriched_ops = manager.attach_turn_metadata_to_operations(operations, turns, conv_id)
    
    # Should keep the original values
    assert enriched_ops[0]["sample_id"] == "conv-99", "Should not overwrite existing sample_id"
    assert enriched_ops[0]["session_id"] == 999, "Should not overwrite existing session_id"
    # But should add missing session_time
    assert enriched_ops[0]["session_time"] == "6:59 pm on 26 August, 2023"
    print(json.dumps(enriched_ops, indent=2))
    print("✓ Existing metadata preserved")
    
    # Test 3: Non-insert operations (should be unchanged)
    print("\n10.3 Don't modify non-insert operations:")
    operations = [
        {
            "operation": "update",
            "memory_id": "abc123",
            "content": "Updated content"
        },
        {
            "operation": "delete",
            "memory_id": "xyz789"
        },
        {
            "operation": "insert",
            "speaker": "John",
            "content": "This should get metadata"
        }
    ]
    
    enriched_ops = manager.attach_turn_metadata_to_operations(operations, turns, conv_id)
    
    # First two should be unchanged
    assert "sample_id" not in enriched_ops[0], "Update op should not get metadata"
    assert "sample_id" not in enriched_ops[1], "Delete op should not get metadata"
    # Third should have metadata
    assert enriched_ops[2]["sample_id"] == "conv-41", "Insert op should get metadata"
    print("✓ Only insert operations modified")
    
    # Test 4: Empty turns list
    print("\n10.4 Handle empty turns list:")
    operations = [
        {
            "operation": "insert",
            "speaker": "John",
            "content": "Test"
        }
    ]
    
    enriched_ops = manager.attach_turn_metadata_to_operations(operations, [], conv_id)
    
    # Should return operations unchanged
    assert "sample_id" not in enriched_ops[0]
    print("✓ Empty turns handled correctly")
    
    # Test 5: Integration with MemoryManager
    print("\n10.5 Integration test with MemoryManager:")
    mem = Memory(embedding_method="openai")
    # manager already created at the start of test
    
    # LLM generates operations without metadata
    llm_operations = [
        {
            "operation": "insert",
            "speaker": "John",
            "content": "Visited NYC, saw major landmarks"
        }
    ]
    
    # Attach metadata from turns
    complete_operations = manager.attach_turn_metadata_to_operations(llm_operations, turns, conv_id)
    
    # Execute through memory manager
    result = manager.execute_batch(mem, complete_operations)
    
    print(f"Batch result: {result['status']}, {result['successful']}/{result['total_commands']} successful")
    assert result['status'] == 'success', "Operation should succeed with complete metadata"
    
    # Verify the inserted memory has correct metadata
    inserted_memory = result['results'][0]['result']['data']
    assert inserted_memory['sample_id'] == "conv-41"
    assert inserted_memory['session_id'] == 9  # Same for all turns in session
    assert inserted_memory['speaker'] == "John"
    print("✓ Integration with MemoryManager works correctly")
    
    print("\n✓ All metadata attachment tests passed!")


def run_all_tests():
    """Run all memory manager tests."""
    print("\n" + "="*70)
    print("  MEMORY MANAGER TEST SUITE (Conversation-Based)")
    print("="*70)
    
    try:
        test_insert_operations()
        test_update_operations()
        test_delete_operations()
        test_search_operations()
        test_batch_operations()
        test_operation_history()
        test_function_schema()
        test_error_handling()
        test_memory_stats()
        test_attach_turn_metadata()
        
        print_section("ALL TESTS COMPLETED")
        print("✓ All tests passed successfully!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
