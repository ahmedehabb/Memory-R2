import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT_QWEN, ANSWER_PROMPT_GPT, ANSWER_PROMPT_LLAMA, ANSWER_PROMPT_LLAMA_BASE
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
            #     self.ANSWER_PROMPT = ANSWER_PROMPT_LLAMA_BASE
            self.ANSWER_PROMPT = ANSWER_PROMPT_LLAMA
        elif self.model == "qwen":
            self.ANSWER_PROMPT = ANSWER_PROMPT_QWEN
        elif self.model == "gpt":
            self.ANSWER_PROMPT = ANSWER_PROMPT_GPT
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

    def answer_question(self, speaker_a_id, speaker_b_id, question, answer, category):
        speaker_a_memories, speaker_a_memory_time= self.search_memory(
            speaker_a_id, question
        )
        speaker_b_memories, speaker_b_memory_time = self.search_memory(
            speaker_b_id, question
        )
        
        search_a_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_a_memories]
        search_b_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_b_memories]

        template = Template(self.ANSWER_PROMPT)
        
        answer_prompt = template.render(
            speaker_1_id=speaker_a_id,
            speaker_2_id=speaker_b_id,
            speaker_1_memories=json.dumps(search_a_memory, indent=4),
            speaker_2_memories=json.dumps(search_b_memory, indent=4),
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
                speaker_a_memories,
                speaker_b_memories,
                speaker_a_memory_time,
                speaker_b_memory_time,
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
                speaker_a_memories,
                speaker_b_memories,
                speaker_a_memory_time,
                speaker_b_memory_time,
                response_time,
            )

    def process_question(self, val, speaker_a_id, speaker_b_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_a_memories,
            speaker_b_memories,
            speaker_a_memory_time,
            speaker_b_memory_time,
            response_time,
        ) = self.answer_question(speaker_a_id, speaker_b_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_a_memories": speaker_a_memories,
            "speaker_b_memories": speaker_b_memories,
            "num_speaker_a_memories": len(speaker_a_memories),
            "num_speaker_b_memories": len(speaker_b_memories),
            "speaker_a_memory_time": speaker_a_memory_time,
            "speaker_b_memory_time": speaker_b_memory_time,
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
            question = item.get("qa", {})["question"]
            answer = item.get("qa", {})["answer"]
            # question_type = item.get("question_type", "")
            # question_date = item.get("question_date", "")
            
            # Create a single user ID for this conversation
            speaker_a_id = item.get("speaker_a")
            speaker_b_id = item.get("speaker_b")

            # Process the single question for this conversation
            question_item = {
                "question": question,
                "answer": answer,
                "category": "none",
                "evidence": [],
                "adversarial_answer": ""
            }
            
            result = self.process_question(question_item, speaker_a_id, speaker_b_id)
            self.results[idx].append(result)

            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(self, qa_list, speaker_a_id, speaker_b_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_id, speaker_b_id)
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
