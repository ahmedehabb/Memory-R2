import argparse
import json
import os
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv
from jinja2 import Template
from prompts import ANSWER_PROMPT_ZEP
from tqdm import tqdm
from zep_cloud import EntityEdge, EntityNode
from zep_cloud.client import Zep

load_dotenv()

# vLLM server configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
# VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")



def make_vllm_request(messages, temperature=0.0, max_tokens=2048):
    """Make a request to the local vLLM server."""
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making vLLM request: {e}")
        return None


TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}

"""


class ZepSearch:
    def __init__(self):
        self.zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.results = defaultdict(list)

    def format_edge_date_range(self, edge: EntityEdge) -> str:
        # return f"{datetime(edge.valid_at).strftime('%Y-%m-%d %H:%M:%S') if edge.valid_at else 'date unknown'} - {(edge.invalid_at.strftime('%Y-%m-%d %H:%M:%S') if edge.invalid_at else 'present')}"
        return f"{edge.valid_at if edge.valid_at else 'date unknown'} - {(edge.invalid_at if edge.invalid_at else 'present')}"

    def compose_search_context(self, edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
        facts = [f"  - {edge.fact} ({self.format_edge_date_range(edge)})" for edge in edges]
        entities = [f"  - {node.name}: {node.summary}" for node in nodes]
        return TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))

    def search_memory(self, run_id, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                edges_results = (
                    self.zep_client.graph.search(
                        user_id=user_id, reranker="cross_encoder", query=query, scope="edges", limit=20
                    )
                ).edges
                node_results = (
                    self.zep_client.graph.search(user_id=user_id, reranker="rrf", query=query, scope="nodes", limit=20)
                ).nodes
                context = self.compose_search_context(edges_results, node_results)
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    return "", 0
                time.sleep(retry_delay)

        end_time = time.time()

        return context, end_time - start_time

    def process_question(self, run_id, user_id, question, answer, category):
        response, search_memory_time, response_time, evidence = self.answer_question(run_id, user_id, question)

        result = {
            "question": question,
            "answer": answer,   
            "category": category,
            "evidence": evidence,
            "response": response,
            "search_memory_time": search_memory_time,
            "response_time": response_time,
        }

        return result

    def answer_question(self, run_id, user_id, question):
        context, search_memory_time = self.search_memory(run_id, user_id, question)

        template = Template(ANSWER_PROMPT_ZEP)
        answer_prompt = template.render(memories=context, question=question)
        print(answer_prompt)
        t1 = time.time()
        response_data = make_vllm_request(
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0
        )
        
        t2 = time.time()
        response_time = t2 - t1
        
        if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
            response_content = response_data["choices"][0]["message"]["content"]
            print("Response: ", response_content)
            return response_content, search_memory_time, response_time, context
        else:
            error_response = "Error: Failed to get response from vLLM server"
            print("Response: ", error_response)
            return error_response, search_memory_time, response_time, context

    def process_data_file(self, file_path, run_id, output_file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            # if idx >= 1:
            #     break
            user_id = f"run_id_{run_id}_experiment_user"
            question = item["question"]
            answer = item["answer"]
            category = item["question_type"]
            result = self.process_question(run_id, user_id, question, answer, category)
            self.results[idx].append(result)

        # Final save at the end
        with open(output_file_path, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    zep_search = ZepSearch()
    zep_search.process_data_file("../../dataset/locomo10.json", args.run_id, "results/zep_search_results.json")
