# Run the experiments
add-openai:
	python run_experiments.py \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_openai\


add-base-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_llama \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model meta-llama/Llama-3.1-8B-Instruct

add-ppo-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_llama \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model <home>/workspace/mem0_ckpt_success/Llama_8b/memory_manager/global_step_15

add-grpo-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_llama \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model <home>/workspace/mem0_ckpt_success/Llama_8b/memory_manager/global_step_8

add-base-qwen:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_qwen \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model Qwen/Qwen2.5-7B-Instruct

pipeline-base-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct

pipeline-base-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct

pipeline-ppo-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_copy \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model <home>/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_70

pipeline-grpo-llama:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_copy \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model <home>/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_30


pipeline-base-qwen:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_qwen \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model Qwen/Qwen2.5-7B-Instruct


pipeline-openai:
	python run_experiments.py \
		--test_type pipeline \
		--model gpt \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_openai \