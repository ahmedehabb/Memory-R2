import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from zep_cloud import Message
from zep_cloud.client import Zep

load_dotenv()


class ZepLongMemEvalAdd:
    def __init__(self, data_path=None):
        self.zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.data_path = data_path
        self.df = None
        if data_path:
            self.load_data()

    def load_data(self):
        """Load the LongMemEval dataset from JSON file."""
        print(f"Loading dataset from {self.data_path}")
        
        if os.path.exists(self.data_path):
            self.df = pd.read_json(self.data_path)
        else:
            # Try parent directory
            parent_path = os.path.join("..", "..", "..", os.path.basename(self.data_path))
            if os.path.exists(parent_path):
                self.df = pd.read_json(parent_path)
            else:
                raise FileNotFoundError(f"Dataset not found at {self.data_path} or {parent_path}")
        
        print(f"Dataset loaded with {len(self.df)} sessions")
        return self.df

    def process_conversation(self, multi_session_idx: int, question_type_filter: Optional[str] = None):
        """Process a single multi-session conversation."""
        # Get session data
        multi_session = self.df["haystack_sessions"].iloc[multi_session_idx]
        multi_session_dates = self.df["haystack_dates"].iloc[multi_session_idx]
        question_type = self.df["question_type"][multi_session_idx]

        # Apply question type filter
        if question_type_filter and question_type != question_type_filter:
            return

        print(f"Processing session {multi_session_idx}: {question_type}")

        try:
            # Create user
            user_id = f"lme_s_experiment_user_{multi_session_idx}"
            self.zep_client.user.add(user_id=user_id)

            # Process each session for this user
            for session_idx, session in enumerate(multi_session):
                session_id = f"lme_s_experiment_session_{multi_session_idx}_{session_idx}"

                # Create Zep session
                self.zep_client.memory.add_session(
                    user_id=user_id,
                    session_id=session_id,
                )

                # Add messages to session
                for msg in tqdm(session, desc=f"Adding messages for session {session_idx}", leave=False):
                    # Parse and format timestamp
                    date = multi_session_dates[session_idx] + " UTC"
                    date_format = "%Y/%m/%d (%a) %H:%M UTC"
                    date_string = datetime.strptime(date, date_format).replace(
                        tzinfo=timezone.utc
                    )

                    # Create message payload
                    message_payload = Message(
                        role=msg["role"],
                        role_type=msg["role"],
                        content=msg["content"],
                        created_at=date_string.isoformat(),
                    )

                    # Add to Zep
                    self.zep_client.memory.add(
                        session_id=session_id,
                        messages=[message_payload],
                    )

        except Exception as e:
            print(f"Error processing session {multi_session_idx}: {e}")

    def process_all_conversations(
        self,
        num_sessions: int = 500,
        question_type_filter: Optional[str] = None,
        start_index: int = 0,
    ):
        """Process all conversations in the dataset."""
        if self.df is None:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        
        # Ensure we don't exceed dataset bounds
        max_sessions = len(self.df)
        end_index = min(start_index + num_sessions, max_sessions)
        actual_sessions = end_index - start_index

        filter_msg = (
            f"question type: {question_type_filter}"
            if question_type_filter
            else "all question types"
        )
        
        print(f"Ingesting {actual_sessions} sessions (indices {start_index}-{end_index - 1}) with {filter_msg}")

        for multi_session_idx in tqdm(range(start_index, end_index), desc="Processing conversations"):
            self.process_conversation(multi_session_idx, question_type_filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dataset/longmemeval_oracle.json", help="Dataset file path")
    parser.add_argument("--num_sessions", type=int, default=500, help="Number of sessions to process")
    parser.add_argument("--question_type", type=str, default=None, help="Filter by question type (default: None)")
    parser.add_argument("--start_index", type=int, default=0, help="Start ingestion from this index (default: 0)")
    args = parser.parse_args()
    
    zep_add = ZepLongMemEvalAdd(data_path=args.dataset)
    zep_add.process_all_conversations(args.num_sessions, args.question_type, args.start_index)
