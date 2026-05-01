import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from jinja2 import Template
import sys
import os
# Add evaluation directory to path for prompts and metrics imports
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)
from prompts import ANSWER_PROMPT_ZEP
from tqdm import tqdm
from zep_cloud import EntityEdge, EntityNode
from zep_cloud.client import Zep

import re

load_dotenv()

# vLLM server configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
# VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

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


CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}

"""


class ZepLongMemEvalSearch:
    def __init__(self, run_id: str):
        self.zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.run_id = run_id
        self.results = {}  # Use dict instead of defaultdict to have control over key format

    def format_edge_date_range(self, edge: EntityEdge) -> str:
        """Format edge date range for display."""
        return f"{edge.valid_at if edge.valid_at else 'date unknown'} - {(edge.invalid_at if edge.invalid_at else 'present')}"

    def compose_search_context(self, edges: List[EntityEdge], nodes: List[EntityNode]) -> str:
        """Compose context from Zep search results."""
        facts = [f"  - {edge.fact} ({self.format_edge_date_range(edge)})" for edge in edges]
        entities = [f"  - {node.name}: {node.summary}" for node in nodes]
        return CONTEXT_TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))

    def search_memory(self, user_id: str, query: str, max_retries=3, retry_delay=1):
        """Search memory using Zep graph search."""
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
                context = self.compose_search_context(edges_results or [], node_results or [])
                break
            except Exception as e:
                print(f"Retrying... Error: {e}")
                retries += 1
                if retries >= max_retries:
                    return "", 0
                time.sleep(retry_delay)

        end_time = time.time()
        return context, end_time - start_time

    def answer_question(self, user_id: str, question: str, question_date: str = None):
        """Generate answer using vLLM with retrieved context."""
        # Format question with date if provided
        formatted_question = f"(date: {question_date}) {question}" if question_date else question
        
        # Search for relevant context
        context, search_memory_time = self.search_memory(user_id, formatted_question)
        
        # Prepare prompt using Jinja2 template
        template = Template(ANSWER_PROMPT_ZEP)
        answer_prompt = template.render(memories=context, question=question)
        
        # Generate answer using vLLM
        t1 = time.time()
        response_data = make_vllm_request(
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        
        if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
            # response_content = response_data["choices"][0]["message"]["content"]
            response_content = answer_extraction(response_data["choices"][0]["message"]["content"])
            return response_content, search_memory_time, response_time, context
        else:
            error_response = "Error: Failed to get response from vLLM server"
            return error_response, search_memory_time, response_time, context

    def process_question(self, df: pd.DataFrame, multi_session_idx: int):
        """Process a single question from the dataset."""
        # Ensure df is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
        
        # Extract question data using iloc for indexing
        if multi_session_idx >= len(df):
            raise IndexError(f"Index {multi_session_idx} out of range for DataFrame with {len(df)} rows")
        
        row = df.iloc[multi_session_idx]
        question_id = row["question_id"]
        question_type = row["question_type"]
        question = row["question"]
        question_date = row["question_date"]
        gold_answer = row["answer"]
        
        user_id = f"lme_s_experiment_user_{multi_session_idx}"
        
        # Generate answer
        response, search_memory_time, response_time, context = self.answer_question(
            user_id, question, question_date
        )
        
        result = {
            "question_id": question_id,
            "question": question,
            "question_date": question_date,
            "question_type": question_type,
            "category": question_type,  # Add category field for compatibility with evals.py
            "answer": gold_answer,
            "response": response,
            "search_memory_time": search_memory_time,
            "response_time": response_time,
            "context": context,
        }
        
        return result

    def process_data_file(self, df_or_path, num_sessions: int = 500, output_file_path: str = "results/zep_longmemeval_results.json", start_index: int = 0):
        """Process the entire dataset.
        
        Args:
            df_or_path: Either a pandas DataFrame or a string path to a JSON file
            num_sessions: Number of sessions to process
            output_file_path: Output file path for results
            start_index: Start index for processing
        """
        # Load DataFrame if path string is provided
        if isinstance(df_or_path, str):
            dataset_path = df_or_path
            print(f"Loading dataset from {dataset_path}")
            
            # Try current directory first
            if os.path.exists(dataset_path):
                file_path = dataset_path
            else:
                # Try parent directory
                parent_path = os.path.join("..", "..", "..", os.path.basename(dataset_path))
                if os.path.exists(parent_path):
                    file_path = parent_path
                else:
                    raise FileNotFoundError(f"Dataset not found at {dataset_path} or {parent_path}")
            
            # Read JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame - data should be a list of dictionaries
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If it's a single dict, wrap it in a list
                df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected data format. Expected list or dict, got {type(data)}")
            
            print(f"Dataset loaded with {len(df)} sessions")
            if len(df) > 0:
                print(f"DataFrame columns: {df.columns.tolist()[:5]}")
                # Verify it's a proper DataFrame
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Failed to create DataFrame. Got {type(df)}")
        else:
            df = df_or_path
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected pd.DataFrame or string path, got {type(df)}")
        
        max_sessions = len(df)
        end_index = min(start_index + num_sessions, max_sessions)
        
        print(f"Processing {end_index - start_index} sessions (indices {start_index}-{end_index - 1})")
        
        for idx in tqdm(range(start_index, end_index), desc="Processing sessions"):
            try:
                result = self.process_question(df, idx)
                # Store as list to match evals.py expected format
                self.results[str(idx)] = [result]
                
                # Save results after each question is processed
                with open(output_file_path, "w") as f:
                    json.dump(self.results, f, indent=4)
            except Exception as e:
                print(f"Error processing session {idx}: {e}")
                continue
        
        # Final save at the end
        with open(output_file_path, "w") as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/longmemeval_s.json", help="Dataset file path")
    parser.add_argument("--num_sessions", type=int, default=500, help="Number of sessions to process")
    parser.add_argument("--output", type=str, default="results/zep_longmemeval_results.json", help="Output file path")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    if os.path.exists(args.dataset):
        df = pd.read_json(args.dataset)
    else:
        # Try parent directory
        parent_path = os.path.join("..", "..", "..", os.path.basename(args.dataset))
        if os.path.exists(parent_path):
            df = pd.read_json(parent_path)
        else:
            raise FileNotFoundError(f"Dataset not found at {args.dataset} or {parent_path}")
    
    print(f"Dataset loaded with {len(df)} sessions")
    
    zep_search = ZepLongMemEvalSearch(run_id=args.run_id)
    zep_search.process_data_file(df, args.num_sessions, args.output, args.start_index)
