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
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for RAG")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks for RAG")

    parser.add_argument("--memory_store_dir", type=str, default="memory_store", help="Dir for ReMA Memory objects")
    parser.add_argument("--embedding_cache_dir", type=str, default=None, help="Dir for ReMA embedding cache")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of concurrent items processed (uses ThreadPoolExecutor; 4-8 recommended for vllm batching)")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/longmemeval_oracle.json",
        help="Path to LongMemEval dataset json (e.g., oracle, s_cleaned, m_cleaned)",
    )

    args = parser.parse_args()
    dataset_tag = os.path.splitext(os.path.basename(args.dataset_path))[0]

    if args.method == "rema_add":
        memory_manager = ReMAADD(
            data_path=args.dataset_path,
            memory_store_dir=args.memory_store_dir,
            memExtractor_url=args.memExtractor_url,
            memExtractor_model=args.memExtractor_model,
            memAgent_url=args.memAgent_url,
            memAgent_model=args.memAgent_model,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        memory_manager.process_all_conversations(max_workers=args.max_workers)
    elif args.method == "rema_search":
        output_file_path = os.path.join(
            args.output_folder,
            f"longmemeval_rema_{dataset_tag}_{args.model}_{args.rl_type}_top{args.top_k}.json"
        )
        memory_searcher = ReMASearch(
            output_path=output_file_path,
            memory_store_dir=args.memory_store_dir,
            answerBot_url=args.answerBot_url,
            answerBot_model=args.answerBot_model,
            top_k=args.top_k,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        memory_searcher.process_data_file(args.dataset_path)
    elif args.method == "add":
        from test_src.memoryr1.add import MemoryADD
        memory_manager = MemoryADD(
            data_path=args.dataset_path,
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
            f"longmemeval_{dataset_tag}_full_{args.model}_{args.test_type}_{args.rl_type}_{args.method}_top{args.top_k}.json"
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
        memory_searcher.process_data_file(args.dataset_path)
    elif args.method == "rag":
        from test_src.rag import RAGManager
        output_file_path = os.path.join(args.output_folder, f"rag_{args.model}_{args.chunk_size}_k{args.num_chunks}.json")
        rag_manager = RAGManager(data_path=args.dataset_path, chunk_size=args.chunk_size, k=args.num_chunks)
        rag_manager.process_all_conversations(output_file_path)
    if args.method == "zep_add":
        from test_src.zep.longmemeval_add import ZepLongMemEvalAdd
        zep_manager = ZepLongMemEvalAdd(data_path=args.dataset_path)
        zep_manager.process_all_conversations(num_sessions=490, question_type_filter=None, start_index=10)
    elif args.method == "zep_search":
        from test_src.zep.longmemeval_search import ZepLongMemEvalSearch
        output_file_path = os.path.join(args.output_folder, f"zep_{args.model}_results_longmemeval_{dataset_tag}.json")
        zep_manager = ZepLongMemEvalSearch(run_id=f"zep_longmemeval_{dataset_tag}")
        zep_manager.process_data_file(args.dataset_path, num_sessions=500, output_file_path=output_file_path, start_index=0)


if __name__ == "__main__":
    main()
