import json
import os
from re import S
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.memory.main import Memory

import threading

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
        conversation = item.get("previous_dialogs", [])
        # haystack_dates = item.get("previous_dialogs_dates", [])
        speaker_a = item.get("speaker_a")
        speaker_b = item.get("speaker_b")
        
        # Process each session with its corresponding timestamp
        for session_idx, session in enumerate(conversation):
            
            messages = []
            messages_reverse = []
            timestamp = session["time_back"]
            chats = session["dialog"]

            for chat in chats:
                if chat["id"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    # print( "current messages", messages[-1])
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                    # print( "current messages reverse", messages_reverse[-1])
                elif chat["id"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    # print( "current messages", messages[-1])
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                    # print( "current messages reverse", messages_reverse[-1])
                else:
                    raise ValueError(f"Unknown speaker: {chat['id']}")

            # print("speaker_a: ", speaker_a, " Messages: ", messages, " timestamp: ", timestamp)
            # print("speaker_b: ", speaker_b, " Messages_reverse: ", messages_reverse, " timestamp: ", timestamp)
            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a, messages, timestamp, f"Adding Memories for {speaker_a}"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b, messages_reverse, timestamp, f"Adding Memories for {speaker_b}"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()
       
        print(f"Messages added successfully for user {speaker_a} and {speaker_b}")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()
