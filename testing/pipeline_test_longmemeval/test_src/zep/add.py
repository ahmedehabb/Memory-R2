import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm
from zep_cloud import Message
from zep_cloud.client import Zep

load_dotenv()


class ZepAdd:
    def __init__(self, data_path=None):
        print("ZEP_API_KEY:", os.getenv("ZEP_API_KEY"))
        self.zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def process_conversation(self, run_id, item, idx, user_id, session_id):
        haystack_sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates", [])

        print("Starting to add memories... for user", idx)
        for session, date in tqdm(zip(haystack_sessions, haystack_dates), desc=f"Processing user {user_id}"):


            for chat in tqdm(session, desc=f"Adding chats for {session_id}", leave=False):
                # self.zep_client.memory.add(
                #     session_id=session_id,
                #     messages=[Message(role="user", role_type="user", content=f"{date}: {chat}")],
                # )
                self.zep_client.graph.add(
                    user_id=user_id,
                    type="text",
                    data=f"{date}: {chat}",
                )


    def process_all_conversations(self, run_id):
        user_id = f"run_id_{run_id}_experiment_user"
        session_id = f"run_id_{run_id}_experiment_session"
        self.zep_client.user.add(user_id=user_id)
        self.zep_client.memory.add_session(user_id=user_id, session_id=session_id)
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        for idx, item in tqdm(enumerate(self.data)):
            self.process_conversation(run_id, item, idx, user_id, session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    zep_add = ZepAdd(data_path="../../dataset/locomo10.json")
    zep_add.process_all_conversations(args.run_id)
