import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT_QWEN, ANSWER_PROMPT_GPT, ANSWER_PROMPT_LLAMA, ANSWER_PROMPT
from tqdm import tqdm

from src.memory.main import Memory
import requests
import re

load_dotenv()

def answer_extraction(text):
    # Pattern 0: <answer>Text</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Pattern 1: **Answer:** Text
    match = re.search(r"\*\*Answer:\*\*\s*(.*)", text)
    if match:
        return match.group(1)

    # Pattern 2: **Answer: Text**
    match = re.search(r"\*\*Answer:\s*(.*?)\*\*", text)
    if match:
        return match.group(1)

    # Pattern 3: Answer: Text
    match = re.search(r"Answer:\s*(.*)", text)
    if match:
        return match.group(1)

    # If none matched
    # raise ValueError("Answer pattern not found")
    return text

class MemorySearch:
    def __init__(self, model="llama", rl_type="base", output_path="results.json", top_k=30, qdrant_path=None, answerBot_url=None, answerBot_model=None):
        self.model = model
        self.mem0_client = Memory(qdrant_path=qdrant_path)
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        if self.model == "llama":
            # if rl_type == "base":
            #     self.ANSWER_PROMPT = ANSWER_PROMPT_LLAMA_BASE
            # else:
            #     self.ANSWER_PROMPT = ANSWER_PROMPT_LLAMA
            self.ANSWER_PROMPT = ANSWER_PROMPT_LLAMA
        elif self.model == "qwen":
            self.ANSWER_PROMPT = ANSWER_PROMPT_QWEN
        elif self.model == "gpt":
            self.ANSWER_PROMPT = ANSWER_PROMPT_GPT
        # self.ANSWER_PROMPT = ANSWER_PROMPT
        self.answerBot_url = answerBot_url
        self.answerBot_model = answerBot_model

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                memories = self.mem0_client.search(
                    query, user_id=user_id, limit=self.top_k
                )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        semantic_memories = [
            {
                "memory": memory["memory"],
                "timestamp": memory["metadata"]["timestamp"],
                "score": round(memory["score"], 2),
            }
            for memory in memories["results"]
        ]
        return semantic_memories, end_time - start_time

    def answer_question(self, user_id, question, answer, category):
        user_memories, user_memory_time = self.search_memory(
            user_id, question
        )

        search_memory = [f"{item['timestamp']}: {item['memory']}" for item in user_memories]

        template = Template(self.ANSWER_PROMPT)
        
        answer_prompt = template.render(
            user_id=user_id.split("_")[0],
            user_memories=json.dumps(search_memory, indent=4),
            question=question,
        )
        messages = [{"role": "system", "content": answer_prompt}]

        if self.model == "gpt":
            t1 = time.time()
            response = self.openai_client.chat.completions.create(
                model=os.getenv("MODEL"), messages=messages, temperature=0.0
            )
            t2 = time.time()
            response_time = t2 - t1
            
            return (
                response.choices[0].message.content,
                user_memories,
                user_memory_time,
                response_time,
            )
        else:        
            t1 = time.time()
            response = requests.post(
                self.answerBot_url,
                json={
                    "model": self.answerBot_model,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0
                }
            )
            
            response = response.json()['choices'][0]['message']['content']
            content = answer_extraction(response)
            
            t2 = time.time()
            response_time = t2 - t1
            return (
                content,
                user_memories,
                user_memory_time,
                response_time,
            )

    def process_question(self, val, user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            user_memories,
            user_memory_time,
            response_time,
        ) = self.answer_question(user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "user_memories": user_memories,
            "num_user_memories": len(user_memories),
            "user_memory_time": user_memory_time,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            # Extract question and answer from the new dataset format
            question = item.get("question", "")
            answer = item.get("answer", "")
            question_type = item.get("question_type", "")
            question_date = item.get("question_date", "")
            
            # Create a single user ID for this conversation
            user_id = f"user_{idx}"

            # Process the single question for this conversation
            question_item = {
                "question": question,
                "answer": answer,
                "category": question_type,
                "evidence": [],
                "adversarial_answer": ""
            }
            
            result = self.process_question(question_item, user_id)
            self.results[idx].append(result)

            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
