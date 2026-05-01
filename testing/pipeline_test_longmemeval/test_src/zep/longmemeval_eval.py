import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# Add evaluation directory to path for metrics imports
eval_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)
from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics


def process_results(input_file_path: str, output_file_path: str = None):
    """Process results and calculate metrics."""
    if output_file_path is None:
        output_file_path = input_file_path.replace(".json", "_evaluated.json")
    
    # Load results
    with open(input_file_path, "r") as f:
        results = json.load(f)
    
    evaluated_results = defaultdict(list)
    all_metrics = defaultdict(list)
    
    print(f"Processing results from {input_file_path}")
    
    # Process each result
    for idx_str, result_data in tqdm(results.items(), desc="Evaluating results"):
        # Handle both list and dict formats
        if isinstance(result_data, list):
            items = result_data
        elif isinstance(result_data, dict):
            # Single result item
            items = [result_data]
        else:
            continue
        
        for item in items:
            question = str(item.get("question", ""))
            gold_answer = str(item.get("answer", ""))
            pred_answer = str(item.get("response", ""))
            question_type = str(item.get("question_type", ""))
            
            # Skip if required fields are missing
            if not question or not gold_answer or not pred_answer:
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(pred_answer, gold_answer)
            bleu_scores = calculate_bleu_scores(pred_answer, gold_answer)
            llm_score = evaluate_llm_judge(question, gold_answer, pred_answer)
            
            # Prepare evaluated result
            evaluated_item = {
                "question": question,
                "question_type": question_type,
                "answer": gold_answer,
                "response": pred_answer,
                "bleu_score": bleu_scores.get("bleu1", 0.0),
                "f1_score": metrics.get("f1", 0.0),
                "exact_match": metrics.get("exact_match", 0),
                "llm_score": llm_score,
                "search_memory_time": item.get("search_memory_time", 0.0),
                "response_time": item.get("response_time", 0.0),
            }
            
            # Add other fields from original result
            for key in ["question_id", "question_date", "context"]:
                if key in item:
                    evaluated_item[key] = item[key]
            
            evaluated_results[idx_str].append(evaluated_item)
            
            # Collect metrics by question type
            if question_type:
                all_metrics[question_type].append({
                    "bleu_score": evaluated_item["bleu_score"],
                    "f1_score": evaluated_item["f1_score"],
                    "exact_match": evaluated_item["exact_match"],
                    "llm_score": evaluated_item["llm_score"],
                })
    
    # Save evaluated results
    with open(output_file_path, "w") as f:
        json.dump(evaluated_results, f, indent=4)
    
    print(f"Evaluated results saved to {output_file_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    # Overall metrics
    all_bleu = []
    all_f1 = []
    all_exact_match = []
    all_llm = []
    
    for question_type, metrics_list in all_metrics.items():
        type_bleu = [m["bleu_score"] for m in metrics_list]
        type_f1 = [m["f1_score"] for m in metrics_list]
        type_exact_match = [m["exact_match"] for m in metrics_list]
        type_llm = [m["llm_score"] for m in metrics_list]
        
        all_bleu.extend(type_bleu)
        all_f1.extend(type_f1)
        all_exact_match.extend(type_exact_match)
        all_llm.extend(type_llm)
        
        print(f"\nQuestion Type: {question_type} ({len(metrics_list)} questions)")
        print(f"  BLEU-1: {np.mean(type_bleu):.4f}")
        print(f"  F1 Score: {np.mean(type_f1):.4f}")
        print(f"  Exact Match: {np.mean(type_exact_match):.4f} ({sum(type_exact_match)}/{len(type_exact_match)})")
        print(f"  LLM Score: {np.mean(type_llm):.4f} ({sum(type_llm)}/{len(type_llm)})")
    
    if all_bleu:
        print("\n" + "-"*50)
        print("Overall Metrics (all question types):")
        print(f"  BLEU-1: {np.mean(all_bleu):.4f}")
        print(f"  F1 Score: {np.mean(all_f1):.4f}")
        print(f"  Exact Match: {np.mean(all_exact_match):.4f} ({sum(all_exact_match)}/{len(all_exact_match)})")
        print(f"  LLM Score: {np.mean(all_llm):.4f} ({sum(all_llm)}/{len(all_llm)})")
    
    return evaluated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zep LongMemEval results")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input results JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output evaluated results JSON file (default: input_file with _evaluated suffix)"
    )
    args = parser.parse_args()
    
    process_results(args.input_file, args.output_file)
