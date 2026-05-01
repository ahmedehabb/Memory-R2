import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.memory.main import Memory

load_dotenv()


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

qdrant_lock = threading.Lock()

class MemoryADD:
    def __init__(self,
        data_path=None,
        batch_size=2,
        test_type="pipeline",
        qdrant_path=None,
        memAgent_url=None,
        memAgent_model=None,
        memExtractor_url=None,
        memExtractor_model=None
        ):
        self.client = Memory(qdrant_path=qdrant_path, memExtractor_url=memExtractor_url, memExtractor_model=memExtractor_model, memAgent_url=memAgent_url, memAgent_model=memAgent_model)

        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                with qdrant_lock:
                    _ = self.client.add(
                        message, user_id=user_id, metadata=metadata
                    )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        # Extract haystack sessions and dates from the new dataset format
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])
        
        # Create a unique user ID for this conversation
        user_id = f"user_{idx}"
        
        # Process each session with its corresponding timestamp
        for session_idx, (session, timestamp) in enumerate(zip(haystack_sessions, haystack_dates)):

            self.add_memories_for_speaker(
                user_id, 
                session, 
                timestamp, 
                f"Adding Memories for User {idx}, Session {session_idx}"
            )

        
        print(f"Messages added successfully for user {idx}")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()
