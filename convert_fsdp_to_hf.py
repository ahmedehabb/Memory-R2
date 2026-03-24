#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.distributed.tensor
import torch
import fire
from glob import glob
from collections import defaultdict


def main(fsdp_checkpoint_path, huggingface_model_path, output_path, world_size=4, attn_implementation="flash_attention_2"):
    state_dict = defaultdict(list)

    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)

    # Prefer non-eager attention for Qwen sliding-window configs to avoid warning/no-op behavior.
    try:
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation)
    except Exception as e:
        print(f"[convert warning] Failed to init with attn_implementation={attn_implementation}: {e}")
        print("[convert warning] Falling back to default model construction.")
        model = AutoModelForCausalLM.from_config(config)

    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)