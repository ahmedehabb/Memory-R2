# Run the experiments
add-base-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_llama_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model meta-llama/Llama-3.1-8B-Instruct


add-sft-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_sft_llama_longmemeval_oracle \
		--memAgent_url http://localhost:4399/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/sft-memAgent-multi-evidence-Llama-3.1-8B-Instruct-0926-2/merged_model/global_step_50


add-ppo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_llama_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-memAgent-multi-evidence-EM-score-copy-Llama-3.1-8B-Instruct-0908/hf_converted/global_step_10


add-grpo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_llama_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-memAgent-multi-evidence-EM-reward-Llama-3.1-8B-Instruct-0909/hf_converted/global_step_8


add-base-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_qwen_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model Qwen/Qwen2.5-7B-Instruct


add-sft-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_sft_qwen_longmemeval_oracle \
		--memAgent_url http://localhost:4399/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/sft-memAgent-multi-evidence-Qwen-7B-Instruct-1001/merged_model/global_step_60


add-ppo-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_qwen_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-memAgent-Qwen2.5-7B-Instruct-multi-evidence-EM-reward-copy-0918/hf_converted/global_step_55


add-grpo-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_qwen_longmemeval_oracle \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-memAgent-multi-evidence-EM-reward-Qwen2.5-7B-Instruct-0913/hf_converted/global_step_14


add-base-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_llama_longbench \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model meta-llama/Llama-3.1-8B-Instruct


add-ppo-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_llama_longbench \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/memory_manager/global_step_15


add-grpo-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_llama_longbench \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/memory_manager/global_step_8


pipeline-base-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_longbench \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


answerAgent-ppo-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type answerAgent \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_longbench \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_70


memAgent-ppo-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type memAgent \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_longbench \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


pipeline-ppo-llama-longbench:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_longbench \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_70


pipeline-base-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct

pipeline-sft-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type sft \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_sft_llama_longmemeval_oracle \
		--answerBot_url http://localhost:4399/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/sft-split118-Llama-3.1-8B-Instruct-0925-1/merged_model/global_step_50

pipeline-ppo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_70


pipeline-grpo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_grpo_llama_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_30


memAgent-ppo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type memAgent \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


memAgent-grpo-llama-longmemeval-oracle:
	python run_experiments.py \
		--model llama \
		--test_type memAgent \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_grpo_llama_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


pipeline-base-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_qwen_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model Qwen/Qwen2.5-7B-Instruct


pipeline-sft-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type sft \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_sft_qwen_longmemeval_oracle \
		--answerBot_url http://localhost:4399/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/sft-split118-Qwen2.5-7B-Instruct-1001-1/merged_model/global_step_60

pipeline-ppo-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_qwen_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-split118-qwen-prompt-em-answer-reward-max-token-2048-Qwen2.5-7B-Instruct-0726-1/hf_converted/global_step_120


pipeline-grpo-qwen-longmemeval-oracle:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_grpo_qwen_longmemeval_oracle \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-split118-qwen-prompt-max-token-2048-agents-5-use-kl-in-reward-Qwen2.5-7B-Instruct-0726-1/hf_converted/global_step_90

run-rag:
	python run_experiments.py \
		--test_type pipeline \
		--method rag \
		--model llama \
		--chunk_size 500 \
		--num_chunks 1 \
		--output_folder results/

run-rag-qwen:
	python run_experiments.py \
		--test_type pipeline \
		--method rag \
		--model qwen \
		--chunk_size 500 \
		--num_chunks 1 \
		--output_folder results/

run-zep-add:
	python run_experiments.py \
		--test_type pipeline \
		--method zep_add \
		--model llama \
		--output_folder results/

run-zep-search:
	python run_experiments.py \
		--test_type pipeline \
		--method zep_search \
		--model qwen \
		--output_folder results/ \
