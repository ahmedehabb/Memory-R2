# Test Pipeline

A comprehensive testing framework for memory-based conversational AI systems, designed to evaluate different memory techniques and retrieval-augmented generation (RAG) approaches.

## Overview

This repository contains a complete pipeline for testing and evaluating memory-enhanced conversational AI systems. It supports multiple memory techniques, different reinforcement learning approaches, and various evaluation metrics to assess the performance of memory-augmented language models.

## Features

- **Multiple Memory Techniques**: Support for pipeline, memAgent, and answerAgent approaches
- **Reinforcement Learning Integration**: Base, PPO, GRPO, and hybrid RL approaches
- **Comprehensive Evaluation**: BLEU scores, F1 scores, and LLM-based evaluation
- **Flexible Configuration**: Configurable models, top-k retrieval, and output paths
- **Parallel Processing**: Multi-threaded evaluation for faster results

## Repository Structure

```
test_pipeline/
├── src/                    # Core source code
│   ├── configs/           # Configuration files
│   ├── embeddings/        # Embedding utilities
│   ├── llms/             # Language model interfaces
│   ├── memory/           # Memory management components
│   ├── utils/            # Utility functions
│   └── vector_stores/    # Vector database interfaces
├── test_src/             # Test-specific implementations
│   ├── memoryr1/         # Memory retrieval implementations
│   │   ├── add.py        # Memory addition logic
│   │   └── search.py     # Memory search logic
│   └── utils.py          # Test utilities and constants
├── dataset/              # Test datasets
│   ├── locomo_test_clean_split118.json
│   ├── locomo_train_clean_split118.json
│   └── locomo_val_clean_split118.json
├── metrics/              # Evaluation metrics
│   ├── llm_judge.py     # LLM-based evaluation
│   └── utils.py         # Metric calculation utilities
├── results/              # Output results directory
├── qdrants/             # Qdrant vector database files
├── run_experiments.py   # Main experiment runner
├── evals.py             # Evaluation script
├── generate_scores.py   # Score generation utility
├── prompts.py           # Prompt templates
├── vllm_server.sh      # VLLM server setup script
└── Makefile            # Build automation
```

## Installation

1. get in the repository:
```bash
cd test_pipeline
```

3. Install dependencies:
```bash
conda create -n memoryr1 python=3.11 -y

conda activate memoryr1

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

4. Set up the vector database:
```bash
# Initialize Qdrant (if not already done)
mkdir -p qdrants
```

## Usage

### Running Experiments

The main entry point is `run_experiments.py` which supports various memory techniques and configurations:

#### Memory Addition
```bash
# Add memories using OpenAI
make add-openai

# Add memories using local model
make add-base
```

#### Memory Search and Evaluation
```bash
# Run pipeline with base RL
make pipeline-base

# Or run directly with custom parameters
python run_experiments.py \
    --test_type pipeline \
    --rl_type base \
    --method search \
    --model llama \
    --top_k 30 \
    --output_folder results/ \
    --qdrant_path qdrants/add_base_copy \
    --answerBot_url http://localhost:8000/v1/chat/completions \
    --answerBot_model meta-llama/Llama-3.1-8B-Instruct
```

### Evaluation

Run comprehensive evaluation on results:

```bash
python evals.py \
python generate_scores.py
```

### Available Options

#### Test Types
- `pipeline`: Standard pipeline approach
- `memAgent`: Memory agent approach
- `answerAgent`: Answer agent approach

#### RL Types
- `base`: Base reinforcement learning
- `ppo`: Proximal Policy Optimization
- `grpo`: Group Relative Policy Optimization
- `ppo_to_grpo`: Hybrid PPO to GRPO
- `grpo_to_ppo`: Hybrid GRPO to PPO

#### Methods
- `add`: Add memories to local qdrant database
- `search`: Search memories and answer from local qdrant database 

#### Models
- `llama`: Baseline model is Llama-3.1-8B-Instruct
- `qwen`: Baseline model is Qwen-2.5-7B-Instruct

## Configuration

### Server Setup

For local model inference, use the provided VLLM server script:

```bash
bash vllm_server.sh
```

### Dataset

The framework uses the LoCoMo dataset for testing. The dataset files are located in the `dataset/` directory:

- `locomo_test_clean_split118.json`: Test dataset
- `locomo_train_clean_split118.json`: Training dataset  
- `locomo_val_clean_split118.json`: Validation dataset

## Output

Results are saved in the `results/` directory with the following naming convention:
```
{model}_{test_type}_{rl_type}_{method}_top{top_k}.json
```

Example: `llama_pipeline_base_search_top30.json`

## Evaluation Metrics

The evaluation system calculates:

- **BLEU Scores**: N-gram overlap between predicted and ground truth answers
- **F1 Scores**: Precision and recall-based evaluation
- **LLM Judge Scores**: AI-powered evaluation of answer quality