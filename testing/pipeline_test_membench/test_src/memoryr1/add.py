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

    def add_memories_for_speaker(self, speaker, messages, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            timestamp = batch_messages[0].get("time", None)
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, key):
        # Extract haystack sessions and dates from the new dataset format
        # haystack_sessions = item.get("haystack_sessions", [])
        # haystack_dates = item.get("haystack_dates", [])
        
        # Create a unique user ID for this conversation
        
        # Process each session with its corresponding timestamp
        for user_idx, conv in enumerate(item):
            user_id = f"user_{key}_{user_idx}"
            sessions = conv.get("message_list", [])
            for session_idx, session in enumerate(sessions):
                messages = []
                for message in session:
                    if key == "Emotion" or key == "Preference":
                        messages.append({"role": "user", "content": f"{user_id}: {message['user']}", "time": message["time"]})
                        messages.append({"role": "assistant", "content": f"assistant: {message['assistant']}", "time": message["time"]})
                    else:    
                        messages.append({"role": "user", "content": f"{user_id}: {message['user_message']}", "time": message["time"]})
                        messages.append({"role": "assistant", "content": f"assistant: {message['assistant_message']}", "time": message["time"]})
                
                self.add_memories_for_speaker(
                    user_id, 
                    messages, 
                    f"Adding Memories for User {user_id}, Session {session_idx}"
                )
        
            print(f"Messages added successfully for user {user_id}")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, key) for key, item in self.data.items()]

            for future in futures:
                future.result()
