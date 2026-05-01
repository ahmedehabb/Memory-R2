import argparse
import os

from test_src.utils import METHODS, TEST_TYPES, RL_TYPES, MODELS
from test_src.rema.add import MemoryADD as ReMAADD
from test_src.rema.search import MemorySearch as ReMASearch

def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--test_type", choices=TEST_TYPES, default="pipeline", help="Memory technique to use")
    parser.add_argument("--rl_type", choices=RL_TYPES, default="base", help="RL type to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--model", choices=MODELS, default="llama", help="Model to use")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--qdrant_path", type=str, default="qdrants/add_base", help="Path to qdrant data")
    parser.add_argument("--memAgent_url", type=str, default=None, help="URL of memAgent")
    parser.add_argument("--memAgent_model", type=str, default=None, help="Model of memAgent")
    parser.add_argument("--memExtractor_url", type=str, default=None, help="URL of memExtractor")
    parser.add_argument("--memExtractor_model", type=str, default=None, help="Model of memExtractor")
    parser.add_argument("--answerBot_url", type=str, default=None, help=" Local answerBot server url")
    parser.add_argument("--answerBot_model", type=str, default=None, help=" Local answerBot model")
    parser.add_argument("--memory_store_dir", type=str, default="memory_store", help="Dir for ReMA Memory objects")
    parser.add_argument("--embedding_cache_dir", type=str, default=None, help="Dir for ReMA embedding cache")

    args = parser.parse_args()

    if args.method == "rema_add":
        memory_manager = ReMAADD(
            data_path="dataset/all_sampled20.json",
            memory_store_dir=args.memory_store_dir,
            memExtractor_url=args.memExtractor_url,
            memExtractor_model=args.memExtractor_model,
            memAgent_url=args.memAgent_url,
            memAgent_model=args.memAgent_model,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        memory_manager.process_all_conversations()
    elif args.method == "rema_search":
        output_file_path = os.path.join(
            args.output_folder,
            f"membench_rema_{args.model}_{args.rl_type}_top{args.top_k}.json"
        )
        memory_searcher = ReMASearch(
            output_path=output_file_path,
            memory_store_dir=args.memory_store_dir,
            answerBot_url=args.answerBot_url,
            answerBot_model=args.answerBot_model,
            top_k=args.top_k,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        memory_searcher.process_data_file("dataset/all_sampled20.json")
    elif args.method == "add":
        from test_src.memoryr1.add import MemoryADD
        memory_manager = MemoryADD(
            data_path="dataset/all_sampled1.json",
            test_type=args.test_type,
            qdrant_path=args.qdrant_path,
            memExtractor_url=args.memExtractor_url,
            memExtractor_model=args.memExtractor_model,
            memAgent_url=args.memAgent_url,
            memAgent_model=args.memAgent_model
        )
        memory_manager.process_all_conversations()
    elif args.method == "search":
        from test_src.memoryr1.search import MemorySearch
        output_file_path = os.path.join(
            args.output_folder,
            f"longbench_{args.model}_{args.test_type}_{args.rl_type}_{args.method}_top{args.top_k}.json"
        )
        memory_searcher = MemorySearch(
            model=args.model,
            rl_type=args.rl_type,
            output_path=output_file_path,
            top_k=args.top_k,
            qdrant_path=args.qdrant_path,
            answerBot_url=args.answerBot_url,
            answerBot_model=args.answerBot_model
        )
        memory_searcher.process_data_file("dataset/all_sampled20.json")


if __name__ == "__main__":
    main()
