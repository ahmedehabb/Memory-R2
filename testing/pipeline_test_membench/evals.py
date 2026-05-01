import argparse
import concurrent.futures
import json
import re
import threading
from collections import defaultdict

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm


def _normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_option_label(pred_answer, choices):
    """
    Extract predicted option label (A/B/C/D) from response text.
    Supports:
    - Explicit label mentions: "Answer: C", "option B"
    - Implicit choice text mentions by matching option strings
    """
    pred_norm = _normalize(pred_answer)

    # Explicit option mention
    m = re.search(r"\b(?:answer\s*[:\-]?\s*|option\s*)?([abcd])\b", pred_norm)
    if m:
        return m.group(1).upper()

    # Fallback: match option text in response
    if isinstance(choices, dict):
        best = None
        for label, option_text in choices.items():
            label = str(label).upper().strip()
            if label not in {"A", "B", "C", "D"}:
                continue
            opt_norm = _normalize(option_text)
            if not opt_norm:
                continue
            if opt_norm in pred_norm:
                score = len(opt_norm)
                if best is None or score > best[0]:
                    best = (score, label)
        if best is not None:
            return best[1]

    return None


def process_item(item_data):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])
        gt_option = str(item.get("ground_truth", "")).strip().upper()
        choices = item.get("choices", {})

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)
        pred_option = _extract_option_label(pred_answer, choices)
        option_acc = 1.0 if (pred_option is not None and pred_option == gt_option) else 0.0

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "ground_truth": gt_option,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
                "pred_option": pred_option,
                "option_acc": option_acc,
            }
        )

    return local_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, default="results/datset100_llama_pipeline_base_search_top30.json", help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation_metrics.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in data.items()]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
