# Run the experiments
add-base-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_llama_msc \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model meta-llama/Llama-3.1-8B-Instruct\
		

add-ppo-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_llama_msc \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-memAgent-multi-evidence-EM-score-copy-Llama-3.1-8B-Instruct-0908/hf_converted/global_step_10


add-grpo-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_llama_msc \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-memAgent-multi-evidence-EM-reward-Llama-3.1-8B-Instruct-0909/hf_converted/global_step_8


pipeline-base-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_llama_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


pipeline-ppo-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_70


pipeline-grpo-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type pipeline \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_grpo_llama_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /home/hk-project-p0022573/lmu_xjh4853/workspace/mem0_ckpt_success/Llama_8b/answer_agent/global_step_30


memAgent-ppo-llama-msc:
	python run_experiments.py \
		--model llama \
		--test_type memAgent \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_llama_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model meta-llama/Llama-3.1-8B-Instruct


add-base-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_base_qwen_msc_second \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model Qwen/Qwen2.5-7B-Instruct \
		2>&1 | tee add_base_qwen_msc.log


add-ppo-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_ppo_qwen_msc \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-memAgent-Qwen2.5-7B-Instruct-multi-evidence-EM-reward-copy-0918/hf_converted/global_step_55 \
		2>&1 | tee add_ppo_qwen_msc.log


add-grpo-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--method add \
		--qdrant_path qdrants/add_grpo_qwen_msc \
		--memAgent_url http://localhost:8000/v1/chat/completions \
		--memAgent_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-memAgent-multi-evidence-EM-reward-Qwen2.5-7B-Instruct-0913/hf_converted/global_step_14 \
		2>&1 | tee add_grpo_qwen_msc.log



pipeline-base-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type base \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_base_qwen_msc_second \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model Qwen/Qwen2.5-7B-Instruct


pipeline-ppo-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type ppo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_ppo_qwen_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/ppo-split118-qwen-prompt-em-answer-reward-max-token-2048-Qwen2.5-7B-Instruct-0726-1/hf_converted/global_step_120


pipeline-grpo-qwen-msc:
	python run_experiments.py \
		--model qwen \
		--test_type pipeline \
		--rl_type grpo \
		--method search \
		--output_folder results/ \
		--top_k 30 \
		--qdrant_path qdrants/add_grpo_qwen_msc \
		--answerBot_url http://localhost:8000/v1/chat/completions \
		--answerBot_model /hkfs/work/workspace/scratch/lmu_xjh4853-mem0rl/grpo-split118-qwen-prompt-max-token-2048-agents-5-use-kl-in-reward-Qwen2.5-7B-Instruct-0726-1/hf_converted/global_step_90

