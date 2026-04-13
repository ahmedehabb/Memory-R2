# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.protocol import collate_fn as data_proto_collate_fn
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.memory_core.prompt_generator import format_turns_for_prompt, generate_memory_prompt_using_facts, format_memory_for_prompt
from verl.rema_trainer.memory.utils.parse_response import extract_llm_json_from_response
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.utils.model import compute_position_id_with_mask
from transformers import PreTrainedTokenizer

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id,
                        prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id,
                                  as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray],
                       repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer,
                 model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (
            not config.enforce_eager and config.free_cache_engine
        ), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens",
                                                 8192)

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert (
            model_hf_config.max_position_embeddings
            >= config.prompt_length + config.response_length
        ), "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.get("enable_sleep_mode", True),
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=config.get('seed', 0)
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
                and self.config.free_cache_engine):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"),
                    non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append({
                    "prompt_token_ids": raw_prompt_ids,
                    "multi_modal_data": multi_modal_data,
                })
        else:
            vllm_inputs = [{
                "prompt_token_ids": raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

            for i in range(len(vllm_inputs)):
                if isinstance(vllm_inputs[i]["prompt_token_ids"], np.ndarray):
                    vllm_inputs[i]["prompt_token_ids"] = vllm_inputs[i][
                        "prompt_token_ids"].tolist()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        is_multi_turn = prompts.meta_info.get("is_multi_turn", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        if is_multi_turn:
            kwargs.update({
                "n": 1, # if is_multi_turn, already repeat in ray_trainer
                "include_stop_str_in_output": True,
                "detokenize": True,
            })
            if prompts.meta_info.get('finish_flag') is not None:
                kwargs['stop'] = [prompts.meta_info['finish_flag']]


        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=
                vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            stop_reasons = []
            gen_response_lengths = []
            text = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    stop_reasons.append(output.outputs[sample_id].finish_reason)
                    gen_response_lengths.append(len(output.outputs[sample_id].token_ids))
                    text.append(output.outputs[sample_id].text)
            
            non_tensor_batch["gen_response_lengths"] = np.array(gen_response_lengths, dtype=object)
            non_tensor_batch["stop_reasons"] = np.array(stop_reasons, dtype=object)
            non_tensor_batch["text"] = np.array(text, dtype=object)
            response = pad_2d_list_to_length(
                response,
                self.pad_token_id,
                max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask,
                                                    self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids,
                                                  self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    raise NotImplementedError("multi_modal_inputs is not supported for multi-turn generation")
                    non_tensor_batch[
                        "multi_modal_inputs"] = _repeat_interleave(
                            non_tensor_batch["multi_modal_inputs"],
                            self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1,
                                         response_length + 1,
                                         device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(
            batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1,
                                                       -1).expand(
                                                           batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response,
                                               eos_token=eos_token_id,
                                               dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask),
                                   dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
                and self.config.free_cache_engine):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def generate_fact_prompts(self, turns: List[List[Dict[str, str]]], current_turn=-1, max_turns=-1) -> List[str]:
        """Generate fact extraction prompts for each conversation based on the provided turns.
        Args:
            turns: List of turns data (each turn is a list of dicts with 'speaker' and 'text')
            current_turn: Current turn index for multi-turn processing. Default is -1 (process all turns).
            max_turns: Maximum number of turns to process in multi-turn setting. Default is -1 (no limit).
        Returns:
            List of fact extraction prompts for each conversation.
        """
        prompts = []
        for turns_data in turns:
            # Parse JSON string if needed
            if isinstance(turns_data, str):
                turns_data = json.loads(turns_data)
                
            if current_turn >= 0 and max_turns > 0:
                # Divide turns across max_turns iterations if needed
                total_turns = len(turns_data)
            
                chunk_size = (total_turns + max_turns - 1) // max_turns  # ceiling division
                start_idx = current_turn * chunk_size
                end_idx = min(start_idx + chunk_size, total_turns)
                
                if start_idx >= total_turns:
                    # No more turns to process - this can happen if max_turns > total_turns
                    # In this case, use empty turns_data (no new dialogue turns to consider)
                    turns_data = []
                    # print(f"No turns to process for iteration {current_turn+1}/{max_turns} (all {total_turns} turns already processed)")
                else:
                    turns_data = turns_data[start_idx:end_idx]
                    # print(f"Generating fact prompt for iteration {current_turn+1}/{max_turns}, processing turns {start_idx}-{end_idx-1} ({len(turns_data)} turns)")

            prompt = (
                "Analyze ONLY the following new dialogue turns and extract new stable facts.\n"
                "The turns are speaker-tagged and already formatted.\n"
                "New turns:\n"
                "```\n{turns}\n```"
            )
            formatted_turns = format_turns_for_prompt(turns_data)
            prompt = prompt.format(turns=formatted_turns)
            prompts.append(prompt)

        return prompts
    
    def generate_memory_prompts(self, sample_ids, chunk_ids, facts_responses, epochs, split, conv_memories=None, rollout_batch_indices=None, snapshot_suffix: str = "") -> Tuple[List[str], List[Memory], MemoryManager]:
        """Load memory snapshots before multi-turn generation and generate memory-augmented prompts for each sample.
        
        Args:
            sample_ids: List of conversation IDs (already filtered to unfinished)
            chunk_ids: List of chunk IDs (already filtered to unfinished)
            facts_responses: List of facts data responses (already filtered to unfinished)
            epochs: List of epochs corresponding to each sample
            split: Data split ('train' or 'validation')
            conv_memories: Optional pre-loaded memories (already filtered to unfinished). If None, will load from cache.
        """
        # Initialize ONE shared memory manager (it's stateless, can be reused!)
        shared_manager = MemoryManager()
        
        # Only load memories if not provided
        if conv_memories is None:
            # Only need to track memories, not managers
            conv_memories: List[Memory] = []
            
            # Load memory states for each conversation-chunk pair from previous chunk (chunk_id - 1)
            for i in range(len(sample_ids)):
                conv_id = sample_ids[i]
                chunk_id = chunk_ids[i]
                epoch = epochs[i]
                index_in_batch = rollout_batch_indices[i]
                
                # Load memory from previous chunk if it exists
                if chunk_id > 1:  # chunk_id starts from 1
                    prev_chunk_id = chunk_id - 1
                    # Always load lineage from normal rollouts.
                    # Inner sampling only namespaces saves to avoid overwriting normal snapshots.
                    loaded_memory = shared_manager.get_snapshot(
                        conv_id,
                        prev_chunk_id,
                        epoch,
                        split,
                        index_in_batch,
                        snapshot_suffix="",
                    )
                    if loaded_memory is not None:
                        conv_memories.append(loaded_memory)
                        # print(f"Loaded memory for conv {conv_id} from chunk {prev_chunk_id}")
                    else:
                        raise Exception(f"No cached memory for conv {conv_id}, chunk {prev_chunk_id}. Required for processing.")
                else:
                    # First chunk, start with empty memory
                    # print(f"Starting fresh memory for conv {conv_id}, chunk {chunk_id}")
                    conv_memories.append(Memory())
        else:
            # print(f"Using {len(conv_memories)} pre-loaded memories")
            pass
        
        # Store prompts for each sample
        prompts = []
        
        # Generate prompts using the memories (either loaded or provided)
        for i in range(len(sample_ids)):
            facts_data = extract_llm_json_from_response(facts_responses[i])
            if not facts_data.get("_parse_success", False):
                facts_data = {"facts": []}
            else:
                # remove the _parse_success key from facts_data
                facts_data.pop("_parse_success", None)

            # print(f"idx: {i} - Generating memory prompt for conv {sample_ids[i]}, with {len(conv_memories)} pre-loaded memories, with {len(facts_data.get('facts', []))} facts")
            prompt = generate_memory_prompt_using_facts(
                conv_memories[i],
                facts_data,
                top_k_memories_for_operations=self.config.top_k_memories_for_operations, 
                similarity_threshold=self.config.similarity_threshold, 
                use_similarity=True
            )

            prompts.append(prompt)

        return prompts, conv_memories, shared_manager

    def generate_single_agent_prompts(
        self,
        sample_ids,
        chunk_ids,
        turns_json_list,
        epochs,
        split,
        conv_memories=None,
        rollout_batch_indices=None,
        snapshot_suffix: str = "",
    ) -> Tuple[List[str], List[Memory], MemoryManager]:
        """Build prompts for the single-agent ablation.

        Skips fact extraction entirely. Retrieves relevant memories using the raw
        dialogue turns as queries and returns a combined prompt (existing memory +
        new turns) for the memory-executor agent.

        Returns the same (prompts, conv_memories, shared_manager) signature as
        generate_memory_prompts so the turn loop can call either transparently.
        """
        shared_manager = MemoryManager()

        if conv_memories is None:
            conv_memories: List[Memory] = []
            for i in range(len(sample_ids)):
                conv_id = sample_ids[i]
                chunk_id = chunk_ids[i]
                epoch = epochs[i]
                index_in_batch = rollout_batch_indices[i]
                if chunk_id > 1:
                    prev_chunk_id = chunk_id - 1
                    loaded_memory = shared_manager.get_snapshot(
                        conv_id, prev_chunk_id, epoch, split, index_in_batch,
                        snapshot_suffix="",
                    )
                    if loaded_memory is not None:
                        conv_memories.append(loaded_memory)
                    else:
                        raise Exception(
                            f"No cached memory for conv {conv_id}, chunk {prev_chunk_id}."
                        )
                else:
                    conv_memories.append(Memory())

        prompts = []
        for i in range(len(sample_ids)):
            turns_data = turns_json_list[i]
            if isinstance(turns_data, str):
                turns_data = json.loads(turns_data)

            formatted_turns = format_turns_for_prompt(turns_data)
            relevant_memories = format_memory_for_prompt(
                conv_memories[i],
                query_turns=turns_data,
                top_k=self.config.top_k_memories_for_operations,
                similarity_threshold=self.config.similarity_threshold,
                use_similarity=True,
            )

            import json as _json
            prompt = (
                "Existing memory:\n"
                "```json\n"
                f"{_json.dumps(relevant_memories, indent=2)}\n"
                "```\n\n"
                "New conversation turns:\n"
                "```json\n"
                f"{_json.dumps(formatted_turns, indent=2)}\n"
                "```"
            )
            prompts.append(prompt)

        return prompts, conv_memories, shared_manager

    @torch.no_grad()
    def multi_turn_generate_sequences(
        self,
        prompts: DataProto,
        tokenizer: PreTrainedTokenizer,
        max_num_turns: int,
        agent_roles: List[str],
        finish_flag: Optional[str],
        system_prompts: Dict[str, str],
        **kwargs,
    ) -> DataProto:
        """Main function responsible for coordinating multi-turn dialogue generation"""
        # Use the parameters directly, assuming they are correctly defined
        # add extra code for multi-turn generation
        # print(f"\n{'='*80}")
        # print(f"MULTI_TURN_GENERATE_SEQUENCES CALLED")
        # print(f"Max num turns: {max_num_turns}")
        # print(f"Agent roles: {agent_roles}")
        # print(f"Finish flag: {finish_flag}")
        # print(f"System prompts keys: {list(system_prompts.keys())}")
        # print(f"rollout indices we got : {prompts.batch['rollout_idx']}...")
        # print(f"{'='*80}\n")
        
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        prompts.meta_info['is_multi_turn'] = True
        prompts.meta_info['finish_flag'] = finish_flag


        sample_ids = prompts.non_tensor_batch["sample_id"]
        batch_size = len(sample_ids)
        # print(f"Sample question [0]: {sample_ids[0][:100]}..." if len(sample_ids) > 0 else "No sample IDs")
        # print(f"Batch size: {batch_size}")
        # Initialize state variables
        history, finish_flags, finish_reason = self._initialize_conversation_state(
            batch_size)
        # print(f"Initialized conversation state: {len(history)} histories, {len(finish_flags)} flags")

        # Multi-turn dialogue generation
        # this will change the history, finish_flags, finish_reason
        # print(f"\nStarting multi-turn conversation...\n")
        latest_outputs, conversation_history, conv_memories, shared_manager, mem_op_stats = self._run_multi_turn_conversation(
            prompts,
            tokenizer=tokenizer,
            max_num_turns=max_num_turns,
            agent_roles=agent_roles,
            system_prompts=system_prompts,
            finish_flag=finish_flag,
            history=history,
            finish_flags=finish_flags,
            finish_reason=finish_reason,
            response_length=self.config.response_length,
            **kwargs,
        )

        # Now for conv_memories we need to save their snapshots after all turns, note that for one sample_id 
        # there can be > 1 conv_memories since we do multiple rollouts for each sample in ppo
        # To avoid overwriting, will save with index in the batch
        # Use rollout indices computed AFTER repeating (not batch_idx which is set before repeating)
        rollout_batch_indices = prompts.batch['rollout_idx'].cpu().numpy().tolist()
        
        snapshot_suffix = prompts.meta_info.get("memory_snapshot_suffix", "")
        for i in range(len(sample_ids)):
            conv_id = sample_ids[i]
            chunk_id = prompts.non_tensor_batch["chunk_id"][i]
            global_idx = rollout_batch_indices[i]

            # print(f"Will save memory snapshot for conv {conv_id}, chunk {chunk_id} at rollout index {global_idx}")
            
            # Save the memory snapshot after all turns for this conversation-chunk pair
            # Use global_idx to ensure unique filenames across all sharded processes
            shared_manager.cache_snapshot(
                conv_memories[i],
                conv_id,
                chunk_id,
                prompts.batch["epoch"][i],
                prompts.meta_info["split"],
                index_in_batch=global_idx,
                snapshot_suffix=snapshot_suffix,
            )
            # print(f"Saved memory snapshot for conv {conv_id}, chunk {chunk_id} at global index {global_idx}")

        # Mark completion reasons
        # this will change the finish_reason
        # print(f"\nMulti-turn conversation completed")
        # print(f"Latest outputs count: {len(latest_outputs)}")
        # print(f"Sample latest output [0]: {latest_outputs[0][:100]}..." if latest_outputs else "No outputs")
        # print(f"Conversation history keys: {list(conversation_history.keys())}")
        # print(f"Finish flags sum: {finish_flags.sum()}/{len(finish_flags)} finished\n")
        
        if max_num_turns > 1:
            self._mark_unfinished_as_max_turns(finish_flags, finish_reason)

        last_round_responses = [{
            m['role']: m['content']
            for m in h[-2:]
        } for h in history]

        # extract information from history record
        num_gen_token_lst = {role: [] for role in agent_roles}
        stop_reason_lst = {role: [] for role in agent_roles}
        for h in history:
            _num_gen_tokens = {role: [] for role in agent_roles}
            _stop_reasons = {role: [] for role in agent_roles}
            for m in h:
                _num_gen_tokens[m['role']].append(m['num_gen_tokens'])
                _stop_reasons[m['role']].append(m['stop_reason'])
            for role in agent_roles:
                num_gen_token_lst[role].append(_num_gen_tokens[role])
                stop_reason_lst[role].append(_stop_reasons[role])

        # print(f"Building tensor dict...")
        # print(f"Last round responses count: {len(last_round_responses)}")
        # print(f"Num gen token list keys: {list(num_gen_token_lst.keys())}")
        # print(f"_stop_reasons : {stop_reason_lst}")
        # print(f"Stop reason list keys: {list(stop_reason_lst.keys())}")
        
        tensor_dict = self._build_tensor_dict(last_round_responses,
                                              conversation_history, 
                                              tokenizer,
                                              num_gen_token_lst,
                                              stop_reason_lst,
                                              max_num_turns,
                                              finish_reason)
        # print(f"Tensor dict built with keys: {list(tensor_dict.keys())}")

        # Prepare return results
        final_output = self._prepare_final_output(
            tensor_dict=tensor_dict,
            latest_outputs=latest_outputs,
            history=history,
            finish_reason=finish_reason,
            agent_roles=agent_roles,
            prompts=prompts,
            conversation_history=conversation_history,
            mem_op_stats=mem_op_stats,
        )

        if self.config.add_checking:
            ###################### TESTING ######################
            # 1. test lengths of history and conversation_history
            #  len(history[i]) == len(conversation_history[role][i]) * len(agent_roles)
            for i in range(len(history)):
                # assert len(history[i]) == len(conversation_history[agent_roles[0]][i]), \
                # Change since now history contains only assistant outputs (8 messages for 4 turns × 2 agents), while conversation_history contains the full conversation with system, user, and assistant messages (9 = 1 system + 4 user + 4 assistant).
                assert len(history[i]) == 2 * (len(conversation_history[agent_roles[0]][i]) - 1) // 2, \
                    f"len(history[i]) = {len(history[i])} != len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])}"
                assert len(conversation_history[agent_roles[0]][i]) == len(conversation_history[agent_roles[1]][i]), \
                    f"len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])} != len(conversation_history[agent_roles[1]][i]) = {len(conversation_history[agent_roles[1]][i])}"

            # 2. check history role name order
            for i in range(len(history)):
                for j in range(len(history[i])):
                    assert history[i][j]['role'] == agent_roles[j % len(agent_roles)], \
                        f"history[i][j]['role'] = {history[i][j]['role']} != agent_roles[j % len(agent_roles)] = {agent_roles[j % len(agent_roles)]}"
                    
                # 2.1 check last round response
                for i_role, role in enumerate(agent_roles):
                    assert history[i][-len(agent_roles) + i_role]['role'] == role, \
                        f"history[i][-len(agent_roles) + i_role]['role'] = {history[i][-len(agent_roles) + i_role]['role']} != role = {role}"
                    assert history[i][-len(agent_roles) + i_role]['content'] == last_round_responses[i][role], \
                        f"history[i][-1]['content'] = {history[i][-1]['content']} != last_round_responses[i][role] = {last_round_responses[i][role]}"
                
            # 3. check conversation_history role name order
            for i_role, role in enumerate(conversation_history.keys()):
                for i in range(len(conversation_history[role])):
                    for j in range(len(conversation_history[role][i])):
                        if j == 0:
                            assert conversation_history[role][i][j]['role'] == "system", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'system'"
                        elif j % 2 == 1:
                            assert conversation_history[role][i][j]['role'] == "user", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'user'"
                        else:
                            assert conversation_history[role][i][j]['role'] == "assistant", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'assistant'"
                            # check history string equals to conversation_string
                            assert conversation_history[role][i][j]['content'] == history[i][i_role + j - 2]['content'], \
                                f"'{[conversation_history[role][i][j]['content']]}' != '{[history[i][i_role + j - 2]['content']]}'"

            # 4. check input_ids
            for i_role, role in enumerate(agent_roles):
                role_tensor_dict = tensor_dict[role]
                for i in range(len(role_tensor_dict["input_ids"])):
                    input_ids = role_tensor_dict["input_ids"][i]
                    labels = role_tensor_dict["labels"][i]
                    attention_mask = role_tensor_dict["attention_mask"][i]
                    step_ids = role_tensor_dict["step_ids"][i]
                    stop_reasons = role_tensor_dict["stop_reasons"][i]
                    num_turn = final_output.non_tensor_batch["num_turns"][i]

                    query_response = tokenizer.decode(input_ids[attention_mask == 1].tolist())
                    raw_query_response = tokenizer.apply_chat_template(
                        conversation_history[role][i], 
                        add_generation_prompt=True, 
                        padding=True, 
                        truncation=False, 
                        max_length=None, 
                        tokenize=False, 
                    ) # + last_round_responses[i][role]
                    
                    assert step_ids.max() == num_turn - 1 or stop_reasons[num_turn - 1] != 0, \
                        f"{step_ids.max()} != {num_turn - 1} or {stop_reasons[num_turn - 1]} != 0"

                    # FIXME: tokenizer has some issues on decode and encode unicode chars.

                    # assert query_response == raw_query_response, \
                    #     f"'{query_response}' != '{raw_query_response}'"
                    # for i_turn in range(num_turn):
                    #     turn_labels = labels[step_ids == i_turn]
                    #     if stop_reasons[i_turn] == 0:
                    #         assert turn_labels[-1] == tokenizer.eos_token_id
                    #         turn_labels = turn_labels[:-1] # drop eos
                    #     response = tokenizer.decode(turn_labels.tolist())
                    #     assert response == history[i][i_role + i_turn * len(agent_roles)]['content'], \
                    #         f"'{response}' != '{history[i][i_role + i_turn * len(agent_roles)]['content']}'"
        

        return final_output

    def _build_tensor_dict(self, 
                           last_round_responses: List[Dict[str, str]],
                           conversation_history: Dict[str, List[List[Dict[str, str]]]],
                           tokenizer: PreTrainedTokenizer,
                           num_gen_token_lst: Dict[str, List[List[int]]],
                           stop_reason_lst: Dict[str, List[List[Optional[str]]]],
                           max_num_turns: int,
                           finish_reason: List[Optional[str]]):
        # conversation_history already contains full conversations with assistant responses
        # (added during generation loop, no need to append here)
        
        input_ids_lst = {role: [] for role in conversation_history.keys()}
        labels_lst = {role: [] for role in conversation_history.keys()}
        step_ids_lst = {role: [] for role in conversation_history.keys()}
        
        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                # encode conversation into input_ids, labels, step_ids
                ####### DEPRECATED NOW ! #######
                # XXX(ziyu): to support reward model
                # input_ids shape is (seq_len + 1, ) if stop reason is 'stop' at
                #  the last turn.
                input_ids, labels, step_ids = encode_conversation(conversation_history[role][i_batch], 
                                                                  tokenizer, 
                                                                  num_gen_token_lst[role][i_batch], 
                                                                  stop_reason_lst[role][i_batch])
                input_ids_lst[role].append(input_ids)
                labels_lst[role].append(labels)
                step_ids_lst[role].append(step_ids)
        
        # Apply padding to create tensors
        batch_size = len(last_round_responses)
        tensor_dict = {}
        finish_reason_array = [] 
        for fr in finish_reason:
            if fr == "reach_max_turn":
                finish_reason_array.append(1)
            elif fr == "completion_token_exceeded":
                finish_reason_array.append(2)
            elif fr == "stop_when_truncated":
                finish_reason_array.append(3)
            elif fr is None:
                finish_reason_array.append(0)
            else:
                raise ValueError(f"Unknown finish reason: {fr}")
        
        
        for role in conversation_history.keys():
            # Find max length for padding
            max_length = max([len(ids) for ids in input_ids_lst[role]])
            # self.config.max_prompt_length + self.config.max_response_length should be = 32768 (whole context length)
            if max_length > self.config.prompt_length + self.config.response_length:
                print(f"WARNING:: role: {role}, max_length={max_length} > {self.config.prompt_length + self.config.response_length}")
                # raise RuntimeError(f"max_length={max_length} > {self.config.max_prompt_length + self.config.max_response_length}")

            # Use max length for padding and gathering
            max_length = self.config.prompt_length + self.config.response_length
            
            # Pad and convert to tensors
            padded_input_ids = torch.full((batch_size, max_length), 
                                          tokenizer.pad_token_id, 
                                          dtype=torch.long)
            padded_labels = torch.full((batch_size, max_length), 
                                      -100,  # IGNORE_INDEX
                                      dtype=torch.long)
            padded_step_ids = torch.full((batch_size, max_length), 
                                          -100,  # IGNORE_INDEX
                                          dtype=torch.long)
            attention_mask = torch.zeros((batch_size, max_length), 
                                       dtype=torch.long)
            
            # Fill in the actual values
            for i, (input_ids, labels, step_ids) in enumerate(zip(
                    input_ids_lst[role], labels_lst[role], step_ids_lst[role])):
                seq_len = min(len(input_ids), max_length)
                padded_input_ids[i, :seq_len] = torch.tensor(input_ids[:seq_len], dtype=torch.long)
                padded_labels[i, :seq_len] = torch.tensor(labels[:seq_len], dtype=torch.long)
                padded_step_ids[i, :seq_len] = torch.tensor(step_ids[:seq_len], dtype=torch.long)
                attention_mask[i, :seq_len] = 1

            # Try decoding here using tokenizer for debugging the first sample for each role
            # decoded_sample = tokenizer.decode(padded_input_ids[0][attention_mask[0] == 1].tolist())
            # print(f"TESTING DECODING -- Role: {role} - Sample decoded input_ids[0]: {decoded_sample}")
            # print(f"TESTING DECODING -- Role: {role} - Sample labels[0]: {padded_labels[0][attention_mask[0] == 1].tolist()}")
            # print(f"TESTING DECODING -- Role: {role} - Sample step_ids[0]: {padded_step_ids[0][attention_mask[0] == 1].tolist()}")

            
            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            padded_num_gen_tokens = torch.full((batch_size, max_num_turns), 
                                              0,
                                              dtype=torch.long)
            for i, num_gen_tokens in enumerate(num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns), 
                                            0,
                                            dtype=torch.bool)

            for i, stop_reasons in enumerate(stop_reason_lst[role]):
                stop_reason_array = np.array([0 if r == 'stop' else 1 for r in stop_reasons])
                padded_stop_reasons[i, :len(stop_reason_array)] = torch.tensor(stop_reason_array, 
                                                                               dtype=torch.bool)

            # Create a separate tensor dict for each role
            tensor_dict[role] = dict(
                {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "step_ids": padded_step_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_gen_tokens": padded_num_gen_tokens,
                    "stop_reasons": padded_stop_reasons,
                    "turn_finished": torch.tensor(finish_reason_array),
                },
            )

        return tensor_dict
    
    def _initialize_conversation_state(self, batch_size):
        """Initialize conversation state variables"""
        history = [[] for _ in range(batch_size)]
        finish_flags = np.zeros(batch_size, dtype=bool)
        finish_reason = [None for _ in range(batch_size)]
        return history, finish_flags, finish_reason

    def _run_multi_turn_conversation(
        self,
        prompts: DataProto,
        tokenizer,
        max_num_turns: int,
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        finish_flag: str,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
        **kwargs,
    ):
        """Execute multi-turn dialogue generation"""
        # print(f"\n_run_multi_turn_conversation called")
        # print(f"len of prompts.non_tensor_batch['turns_json']: {len(prompts.non_tensor_batch['turns_json'])}")
        
        # Initialize arrays - fact_prompts will be generated inside the turn loop
        batch_size = len(prompts.non_tensor_batch['turns_json'])
        fact_prompts = np.array(["" for _ in range(batch_size)], dtype=object)
        executor_prompts = np.array(["" for _ in range(batch_size)], dtype=object)
        
        # Initialize conversation memories with None since at first turn we need to load from cache
        conv_memories = None
        
        # Initialize operation statistics storage for full batch size
        mem_op_stats = {
            'insert_successful': [0] * batch_size,
            'delete_successful': [0] * batch_size,
            'update_successful': [0] * batch_size,
            'insert_total': [0] * batch_size,
            'delete_total': [0] * batch_size,
            'update_total': [0] * batch_size,
            'dia_ids_affected_per_turn': [[] for _ in range(batch_size)],  # Track dia_ids affected per turn
        }
        
        assert len(finish_flags) == len(fact_prompts), f"{finish_flags.shape} != {len(fact_prompts)}"

        conversation_history = {
            role: [None for _ in range(len(fact_prompts))]
            for role in agent_roles
        }
        # print(f"Initialized conversation_history for roles: {list(conversation_history.keys())}")

        for i_turn in range(max_num_turns):
            # Get indices of unfinished samples
            unfinished_indices = np.where(~finish_flags)[0]
            # print(f"\n{'='*60}")
            # print(
            #     f"TURN {i_turn+1}/{max_num_turns}: {len(unfinished_indices)}/{len(fact_prompts)} unfinished"
            # )
            # print(f"{'='*60}")

            if len(unfinished_indices) == 0:
                # print("All samples finished, breaking out of turn loop.")
                break
            
            single_agent_mode = getattr(self.config, 'single_agent_mode', False)

            if single_agent_mode:
                # Single-agent ablation: skip meta-thinking entirely.
                # Build executor prompts directly from raw turns + memory state.
                unfinished_sample_ids = [prompts.non_tensor_batch["sample_id"][idx] for idx in unfinished_indices]
                unfinished_chunk_ids = [prompts.non_tensor_batch["chunk_id"][idx] for idx in unfinished_indices]
                load_rollout_indices = prompts.batch["source_rollout_idx"] if "source_rollout_idx" in prompts.batch.keys() else prompts.batch["rollout_idx"]
                unfinished_rollout_batch_indices = [load_rollout_indices[idx] for idx in unfinished_indices]
                unfinished_epochs = [prompts.batch["epoch"][idx] for idx in unfinished_indices]
                unfinished_turns_json = [prompts.non_tensor_batch["turns_json"][idx] for idx in unfinished_indices]
                unfinished_conv_memories = None
                if conv_memories is not None:
                    unfinished_conv_memories = [conv_memories[idx] for idx in unfinished_indices]

                # Slice turns to current turn chunk (same chunking logic as generate_fact_prompts)
                sliced_turns_json = []
                for turns_data in unfinished_turns_json:
                    if isinstance(turns_data, str):
                        turns_data = json.loads(turns_data)
                    total_turns = len(turns_data)
                    chunk_size = (total_turns + max_num_turns - 1) // max_num_turns
                    start_idx = i_turn * chunk_size
                    end_idx = min(start_idx + chunk_size, total_turns)
                    sliced_turns_json.append(turns_data[start_idx:end_idx] if start_idx < total_turns else [])

                sa_prompts, updated_conv_memories, shared_manager = self.generate_single_agent_prompts(
                    sample_ids=unfinished_sample_ids,
                    chunk_ids=unfinished_chunk_ids,
                    turns_json_list=sliced_turns_json,
                    epochs=unfinished_epochs,
                    split=prompts.meta_info["split"],
                    conv_memories=unfinished_conv_memories,
                    rollout_batch_indices=unfinished_rollout_batch_indices,
                    snapshot_suffix=prompts.meta_info.get("memory_snapshot_suffix", ""),
                )

                if conv_memories is None:
                    conv_memories = [None] * len(prompts.non_tensor_batch["sample_id"])
                for i, idx in enumerate(unfinished_indices):
                    conv_memories[idx] = updated_conv_memories[i]
                    executor_prompts[idx] = sa_prompts[i]
            else:
                # Two-agent mode: regenerate fact prompts for the current turn chunk
                fact_prompts = self.generate_fact_prompts(
                    prompts.non_tensor_batch["turns_json"],
                    current_turn=i_turn,
                    max_turns=max_num_turns
                )
                fact_prompts = np.array(fact_prompts, dtype=object)

            # Each role takes turns generating in every round
            for i_role, role in enumerate(agent_roles):
                # In single-agent mode, skip the meta-thinking (agent_roles[0]) generation.
                # Instead of a bare continue, inject dummy entries so that:
                #   - conversation_history['meta_thinking'] is non-None (encode_conversation needs it)
                #   - history has a meta_thinking entry at every turn (assertions + [-2] lookup need it)
                #   - stop_reason='completion_token_exceeded' ensures labels are all IGNORE_INDEX (no gradient)
                if single_agent_mode and role == agent_roles[0]:
                    for _idx in unfinished_indices:
                        if conversation_history[agent_roles[0]][_idx] is None:
                            conversation_history[agent_roles[0]][_idx] = [
                                {"role": "system", "content": system_prompts.get(agent_roles[0], "")}
                            ]
                        # Append empty user+assistant pair — matches what _prepare_role_prompts produces
                        conversation_history[agent_roles[0]][_idx].append({"role": "user", "content": ""})
                        conversation_history[agent_roles[0]][_idx].append({"role": "assistant", "content": ""})
                        # Dummy history entry: completion_token_exceeded → all labels IGNORE_INDEX, zero gradient
                        history[_idx].append({
                            "role": agent_roles[0],
                            "content": "",
                            "num_gen_tokens": 0,
                            "stop_reason": "completion_token_exceeded",
                        })
                    continue

                # Choose questions based on role
                if role == agent_roles[0]:
                    questions = fact_prompts
                else:
                    questions = executor_prompts

                prompt_proto, chat_lst = self._prepare_role_prompts(
                    role,
                    unfinished_indices,
                    history,
                    questions,
                    agent_roles,
                    system_prompts,
                    tokenizer,
                    conversation_history,
                )

                # check current state length
                non_trunc_input = tokenizer.apply_chat_template(
                    chat_lst,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=False,
                    max_length=None,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                )
                # state length
                seq_lens = non_trunc_input["attention_mask"].sum(dim=1).tolist()
                # print(f"  Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, prompt_length={self.config.prompt_length}")
                # if state length is larger than prompt length, the trajectory is terminated 
                if not all([l <= self.config.prompt_length for l in seq_lens]):
                    # print(f"  WARNING: Some sequences exceed prompt_length!")
                    # drop the terminated trajectories
                    new_seq_lens = []
                    new_unfinished_indices = []
                    new_prompt_protos = []
                    new_chat_lst = []
                    for i, idx in enumerate(unfinished_indices):
                        if seq_lens[i] <= self.config.prompt_length:
                            new_unfinished_indices.append(idx)
                            new_prompt_protos.append(prompt_proto[i])
                            new_seq_lens.append(seq_lens[i])
                            new_chat_lst.append(chat_lst[i])
                        else:
                            # set finish flag and finish reason 
                            finish_flags[idx] = True
                            finish_reason[idx] = "completion_token_exceeded"
                            # print(f'idx={idx}, completion_token_exceeded')
                            # if the next gen is for reasoning agent, we need to add a dummy response in history
                            if role == agent_roles[1]:
                                history[idx].append(
                                    {"role": agent_roles[1], "content": "", 
                                    "num_gen_tokens": 0, 
                                    "stop_reason": "completion_token_exceeded"}
                                )
                                # update conversation history for reasoning agent
                                # TODO:: added dummy response for assistant in chat_lst so conversation history is of same lengths
                                chat_lst[i].append({"role": "assistant", "content": ""})
                                conversation_history[agent_roles[1]][idx] = chat_lst[i]
                            else:
                                if i_turn == 0:
                                    raise RuntimeError(f"1st round prompt larger than prompt length: {seq_lens[i]} > {self.config.prompt_length}")

                    # update unfinished_indices
                    unfinished_indices = new_unfinished_indices
                    if len(unfinished_indices) == 0:
                        break

                    # collate prompt needed to generate this round
                    prompt_proto = data_proto_collate_fn(new_prompt_protos)
                    seq_lens = new_seq_lens
                    chat_lst = new_chat_lst
                
                prompt_proto.meta_info.update(prompts.meta_info)

                # Generate responses for current role
                # print(f"  Generating responses...")
                current_outputs, num_gen_tokens, stop_reasons, resp_lens = self._generate_role_responses(
                    prompt_proto, tokenizer, response_length, **kwargs)
                if not (
                    len(current_outputs) == len(unfinished_indices)
                    and len(num_gen_tokens) == len(unfinished_indices)
                    and len(stop_reasons) == len(unfinished_indices)
                ):
                    raise RuntimeError(
                        "Mismatch between unfinished indices and generation outputs: "
                        f"unfinished={len(unfinished_indices)}, outputs={len(current_outputs)}, "
                        f"num_gen_tokens={len(num_gen_tokens)}, stop_reasons={len(stop_reasons)}"
                    )
                # print(f"  Generated {len(current_outputs)} responses")
                # print(f"  Sample output [0]: {current_outputs[0]}..." if current_outputs else "  No outputs")
                # print(f"  Sample num_gen_tokens [0]: {num_gen_tokens[0]}" if num_gen_tokens else "  No tokens")
                # print(f"  Sample stop_reason [0]: {stop_reasons[0]}" if stop_reasons else "  No stop reasons")

                if role == agent_roles[0]:
                    # Update executor questions for next role
                    # print(f"  Updating executor questions for next role...")
                    extracted_facts = [current_outputs[i] for i in range(len(current_outputs))]
                    # print(f"  Extracted facts for {len(extracted_facts)} samples, \nfacts:: {extracted_facts}")
                    
                    # Filter ALL inputs to unfinished samples
                    unfinished_sample_ids = [prompts.non_tensor_batch["sample_id"][idx] for idx in unfinished_indices]
                    unfinished_chunk_ids = [prompts.non_tensor_batch["chunk_id"][idx] for idx in unfinished_indices]
                    # source_rollout_idx is the index of the sample in the original rollout batch (before filtering), 
                    # which is needed for loading the correct memory snapshots. If source_rollout_idx is not available, fallback to rollout_idx which should be the same in this context since we are not doing any shuffling or reordering.
                    load_rollout_indices = prompts.batch["source_rollout_idx"] if "source_rollout_idx" in prompts.batch.keys() else prompts.batch["rollout_idx"]
                    unfinished_rollout_batch_indices = [load_rollout_indices[idx] for idx in unfinished_indices]
                    unfinished_epochs = [prompts.batch["epoch"][idx] for idx in unfinished_indices]
                    unfinished_conv_memories = None
                    if conv_memories is not None:
                        unfinished_conv_memories = [conv_memories[idx] for idx in unfinished_indices]
                    
                    # Call with consistently filtered inputs
                    memory_prompts, updated_conv_memories, shared_manager = self.generate_memory_prompts(
                        sample_ids=unfinished_sample_ids,
                        chunk_ids=unfinished_chunk_ids,
                        facts_responses=extracted_facts,
                        epochs=unfinished_epochs,
                        split=prompts.meta_info["split"],
                        conv_memories=unfinished_conv_memories,
                        rollout_batch_indices=unfinished_rollout_batch_indices,
                        snapshot_suffix=prompts.meta_info.get("memory_snapshot_suffix", ""),
                    )
                    
                    # Rebuild full arrays for next iteration
                    if conv_memories is None:
                        # First turn - initialize with full batch size
                        conv_memories = [None] * len(prompts.non_tensor_batch["sample_id"])
                        for i, idx in enumerate(unfinished_indices):
                            conv_memories[idx] = updated_conv_memories[i]
                    else:
                        # Update only unfinished samples
                        for i, idx in enumerate(unfinished_indices):
                            conv_memories[idx] = updated_conv_memories[i]
                    
                    # Update executor_prompts at unfinished indices only
                    for i, idx in enumerate(unfinished_indices):
                        executor_prompts[idx] = memory_prompts[i]

                elif role == agent_roles[1]:
                    # Execute memory operations after the first agent's turn
                    # print(f"  Executing memory operations for {len(current_outputs)} samples...")
                    rewards, ops_per_sample, batch_op_stats = self._execute_memory_operations(
                        prompts, conv_memories, shared_manager, current_outputs, unfinished_indices, i_turn
                    )
                    if len(conv_memories) != len(mem_op_stats['insert_successful']):
                        raise RuntimeError(
                            f"conv_memories length mismatch: {len(conv_memories)} != {len(mem_op_stats['insert_successful'])}"
                        )
                    # Accumulate operation statistics for unfinished samples
                    for i, idx in enumerate(unfinished_indices):
                        for key in mem_op_stats.keys():
                            if key != 'dia_ids_affected_per_turn':  # Skip this one, handled separately
                                mem_op_stats[key][idx] += batch_op_stats[key][idx]
                    
                    # For dia_ids_affected_per_turn, extend the list instead of adding
                    for i, idx in enumerate(unfinished_indices):
                        mem_op_stats['dia_ids_affected_per_turn'][idx].extend(batch_op_stats['dia_ids_affected_per_turn'][idx])
                
                # XXX(ziyu): remove finish flag in output for reasoning agent here
                #  consider move to a post-processing function
                if role == agent_roles[1] and finish_flag:
                    current_outputs = [
                        output.replace(finish_flag, "").rstrip() for output in current_outputs
                    ]
                
                # Append assistant responses to chat_lst and store complete conversation
                for i, output in enumerate(current_outputs):
                    chat_lst[i].append({"role": "assistant", "content": output})
                    idx = unfinished_indices[i]
                    conversation_history[role][idx] = chat_lst[i]

                # XXX(ziyu): side effect on `history`
                # print(f"  Updating history and checking finish conditions...")
                self._update_history_and_check_finish(
                    role,
                    current_outputs,
                    unfinished_indices,
                    history,
                    finish_flags,
                    finish_reason,
                    finish_flag,
                    agent_roles,
                    num_gen_tokens,
                    stop_reasons,
                    fact_prompts,
                    executor_prompts,
                    conversation_history,
                    system_prompts,
                    tokenizer,
                )
                unfinished_indices = np.where(~finish_flags)[0]
                # print(f"  After update: {len(unfinished_indices)} still unfinished, {finish_flags.sum()} finished")
                if len(unfinished_indices) == 0:
                    # print(f"  All samples finished, breaking from role loop")
                    break
            
        # use the last output of each agent as latest output response
        latest_outputs = [h[-1]['content'] for h in history]

        return latest_outputs, conversation_history, conv_memories, shared_manager, mem_op_stats
    
    def _execute_memory_operations(
        self,
        prompt_data: DataProto,
        memories: List[Memory],
        shared_manager: MemoryManager,
        response_texts,
        unfinished_indices: List[int] = None,
        current_turn_id: int = 0,
    ) -> tuple[list[float], list[list[dict]], dict]:
        """Execute memory operations from batch responses without caching.
        
        Args:
            prompt_data: DataProto containing prompt metadata.
            memories: List of Memory objects.
            shared_manager: MemoryManager instance shared across memories.
            response_texts: List of response strings from the model (only for unfinished samples).
            unfinished_indices: List of indices that are still being processed. If None, assumes all samples.

        Returns:
            list: Rewards for each conversation based on operation execution success.
                  Reward = json_correctness * operation_success_rate
                  - If JSON invalid: reward = 0.0
                  - If JSON valid but 0 ops: reward = 1.0 * 1.0 = 1.0 (intentional no-ops)
                  - If JSON valid with N ops, M successful: reward = 1.0 * (M/N)
            memory_operations_per_sample: List of lists of dicts - recorded operations per sample.
            operation_stats: Dict with per-sample operation statistics for each type.
        """
        if memories is None:
            raise RuntimeError("memories must be initialized before executing memory operations")

        rewards = []
        # Per-sample recorded operations (parsed from LLM)
        memory_operations_per_sample: list[list[dict]] = [[] for _ in range(len(memories))]
        # Per-sample operation statistics
        operation_stats = {
            'insert_successful': [0] * len(memories),
            'delete_successful': [0] * len(memories),
            'update_successful': [0] * len(memories),
            'insert_total': [0] * len(memories),
            'delete_total': [0] * len(memories),
            'update_total': [0] * len(memories),
            'dia_ids_affected_per_turn': [[] for _ in range(len(memories))],  # Track dia_ids affected in each turn
        }
        sample_ids = prompt_data.non_tensor_batch["sample_id"]
        chunk_ids = prompt_data.non_tensor_batch["chunk_id"]
        turns_json = prompt_data.non_tensor_batch["turns_json"]
        
        # If unfinished_indices not provided, assume all samples
        if unfinished_indices is None:
            unfinished_indices = list(range(len(memories)))

        if len(response_texts) != len(unfinished_indices):
            raise RuntimeError(
                "response_texts must align with unfinished_indices: "
                f"response_texts={len(response_texts)}, unfinished_indices={len(unfinished_indices)}"
            )

        if len(set(unfinished_indices)) != len(unfinished_indices):
            raise RuntimeError("unfinished_indices contains duplicate indices")

        for idx in unfinished_indices:
            if idx < 0 or idx >= len(memories):
                raise RuntimeError(
                    f"unfinished index out of range: idx={idx}, valid_range=[0, {len(memories) - 1}]"
                )
        
        # Create mapping from unfinished index to response_texts index
        idx_to_response_idx = {idx: i for i, idx in enumerate(unfinished_indices)}
        
        for idx, memory in enumerate(memories):
            # Skip if this sample is not in unfinished_indices (already finished)
            if idx not in idx_to_response_idx:
                # For finished samples, give neutral reward and empty operations
                rewards.append(0.0)
                memory_operations_per_sample[idx] = []
                continue
            
            response_idx = idx_to_response_idx[idx]
            response_text = response_texts[response_idx]
            # Parse operations from response
            response_json = extract_llm_json_from_response(response_text)
            json_parse_success = response_json.get("_parse_success", False)
            operations = response_json.get("operations", [])
            
            # If JSON parsing failed, reward is 0
            if not json_parse_success:
                rewards.append(0.0)
                # print(f"Conv {sample_ids[idx]}, chunk {chunk_ids[idx]}: "
                #       f"JSON parsing FAILED - Reward=0.0")
                continue
            
            # Attach turn metadata
            turns = json.loads(turns_json[idx])
            operations = shared_manager.attach_turn_metadata_to_operations(
                operations, turns, sample_ids[idx]
            )

            # Record the parsed operations for downstream evaluation / judging
            memory_operations_per_sample[idx] = operations # list of dicts

            # Execute operations (even if empty list)
            result = shared_manager.execute_batch(memories[idx], operations)
            
            total_ops = result.get("total_commands", 0)
            successful_ops = result.get("successful", 0)
            
            # Extract per-type statistics from result
            operation_stats['insert_successful'][idx] = result.get('insert_successful', 0)
            operation_stats['delete_successful'][idx] = result.get('delete_successful', 0)
            operation_stats['update_successful'][idx] = result.get('update_successful', 0)
            operation_stats['insert_total'][idx] = result.get('insert_total', 0)
            operation_stats['delete_total'][idx] = result.get('delete_total', 0)
            operation_stats['update_total'][idx] = result.get('update_total', 0)
            
            # Track dia_ids affected by successful operations in each turn
            dia_ids_per_turn = {}
            
            for op_idx, op_result in enumerate(result.get('results', [])):
                dia_id = op_result.get('command_dia_id')
                if dia_id:
                    if current_turn_id not in dia_ids_per_turn:
                        dia_ids_per_turn[current_turn_id] = set()
                    dia_ids_per_turn[current_turn_id].add(dia_id)
                #     print(f"[_execute_memory_operations]   Recorded: turn_id={turn_id}, dia_id={dia_id}")
                # else:
                #     print(f"[_execute_memory_operations]   Warning: dia_id={dia_id} is missing from command")
            
            # Convert to list format: [(turn_id, [dia_ids])]
            operation_stats['dia_ids_affected_per_turn'][idx] = [
                {'turn_id': turn_id, 'dia_ids': list(dia_ids)}
                for turn_id, dia_ids in sorted(dia_ids_per_turn.items())
            ]
            # print(f"[_execute_memory_operations] Conv {sample_ids[idx]}: Stored dia_ids_affected_per_turn = {operation_stats['dia_ids_affected_per_turn'][idx]}")
            
            # Calculate operation success rate
            if total_ops == 0:
                # No operations - intentional, so 100% success
                ops_reward = 1.0
            else:
                ops_reward = successful_ops / total_ops
            
            # Final reward: JSON correct (1.0) * operation success rate
            final_reward = 1.0 * ops_reward
            rewards.append(final_reward)
            
            # print(f"Conv {sample_ids[idx]}, chunk {chunk_ids[idx]}: "
            #       f"JSON=OK, Ops={successful_ops}/{total_ops}, Format Reward={final_reward:.3f}")
            
            if result["status"] not in ["success", "partial"]:
                print(f"Warning: Memory operations had issues: {result}")
        
        return rewards, memory_operations_per_sample, operation_stats


    def _prepare_role_prompts(
        self,
        role: str,
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        questions: List[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        tokenizer,
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
    ) -> Tuple[DataProto, List[List[Dict[str, str]]]]:
        """Prepare prompts for a specific role"""
        # print(f"    _prepare_role_prompts called for role={role}")
        # print(f"      Unfinished indices count: {len(unfinished_indices)}")
        # print(f"      Unfinished indices: {unfinished_indices.tolist() if hasattr(unfinished_indices, 'tolist') else unfinished_indices}")

        # Prepare history and questions for currently unfinished samples
        current_history = [history[idx] for idx in unfinished_indices]
        current_questions = [questions[idx] for idx in unfinished_indices]
        # print(f"      Current history lengths: {[len(h) for h in current_history]}")

        # Build chat list
        # print(f"      Building chat list for role...")
        # Get existing conversations for unfinished samples
        existing_convs = [conversation_history[role][idx] for idx in unfinished_indices]
        chat_lst = self._build_chat_list_for_role(
            role,
            current_history,
            current_questions,
            system_prompts,
            agent_roles,
            existing_convs,
        )
        # print(f"      Chat list built, count: {len(chat_lst)}")

        # Apply chat template and encode
        # print(f"      Applying chat template...")
        inputs = self._apply_chat_template(chat_lst, tokenizer)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # print(f"      Input IDs shape: {input_ids.shape}")
        # print(f"      Attention mask shape: {attention_mask.shape}")

        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        # print(f"      DataProto created successfully")
        return data, chat_lst

    def _build_chat_list_for_role(
        self,
        role: str,
        history_list: List[List[Dict[str, str]]],
        questions: List[str],
        system_prompts: Dict[str, str],
        agent_roles: List[str],
        existing_conversations: List[List[Dict[str, str]]] = None,
    ):
        """Build chat list for a specific role by extending existing conversation with new question"""
        # print(f"        _build_chat_list_for_role called for role={role}")
        # print(f"          History list length: {len(history_list)}")
        # print(f"          Questions count: {len(questions)}")

        chat_lst = []
        
        for i, question in enumerate(questions):
            # If existing conversation provided, extend it; otherwise start fresh
            first_turn = False
            if existing_conversations and existing_conversations[i] is not None:
                # Copy existing conversation and append new question
                chat = existing_conversations[i].copy()
                # print(f"          Sample {i}: Extending existing conversation with {len(chat)} messages")
            else:
                # First turn - start with system prompt only
                first_turn = True
                chat = [{
                    "role": "system",
                    "content": system_prompts[role]
                }]
                # print(f"          Sample {i}: Starting fresh conversation")
            
            # Append the new question
            if first_turn:
                chat.append({"role": "user", "content": question})
            else:
                if role == agent_roles[0]:
                    chat.append({
                        "role": "user",
                        "content": (
                            "Previous session turns (user and assistant) are provided for context only. "
                            "You can refer to them for disambiguation or clarity, but do NOT treat them as new events.\n"
                            + question
                        )
                    })
                else:
                    chat.append({
                        "role": "user",
                        "content": (
                            "Previous memory actions and context are provided for reference only. "
                            "You may refer to them for disambiguation or continuity, but do NOT treat them as new facts or perform duplicate actions. "
                            "Based on the extracted facts below, perform the required memory operations.\n"
                            + question
                        )
                    })

            # if role == agent_roles[0]:
            #     print(f"meta-thinking agent:: {chat[-1]}...\n")
            # else:
            #     print(f"reasoning agent:: {chat[-1]}...\n")
            
            chat_lst.append(chat)
        
        # print(f"          Built {len(chat_lst)} chat sequences")
        # if len(chat_lst) > 0:
            # print(f"          Sample chat [0] length: {len(chat_lst[0])} messages")
            # print(f"          Sample chat [0]: {chat_lst[0]}")

        return chat_lst

    def _apply_chat_template(self, chat_lst: List[List[Dict[str, str]]],
                             tokenizer):
        """Apply chat template and encode"""
        # print(f"        _apply_chat_template called")
        # print(f"          Processing {len(chat_lst)} chat sequences")
        # print(f"          Max length: {self.config.prompt_length}")
        
        result = tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        # print(f"          Template applied, result keys: {list(result.keys())}")
        return result

    def _generate_role_responses(self, prompt_proto: DataProto, tokenizer,
                                 response_length: int, **kwargs):
        """Generate responses for the current role"""
        # print(f"    _generate_role_responses called")
        # print(f"      Response length: {response_length}")
        # print(f"      Prompt proto batch keys: {list(prompt_proto.batch.keys())}")
        
        output = self.generate_sequences(prompt_proto, **kwargs)
        # print(f"      Generation completed")
        
        resp_lens = output.batch['attention_mask'][:, -response_length:].sum(dim=1).tolist()
        vllm_output_text = output.non_tensor_batch['text'].tolist()
        # print(f"      Response lengths: {resp_lens}")
        # output_text = tokenizer.batch_decode(
        #     output.batch["input_ids"][:, -response_length:],
        #     skip_special_tokens=False,
        # )

        # # Remove padding and EOS tokens from the output in one pass
        # pad_token = tokenizer.pad_token
        # eos_token = tokenizer.eos_token
        # output_text_clean = [
        #     text.replace(pad_token, "").replace(eos_token, "")
        #     for text in output_text
        # ]

        # for i, (decode_txt, vllm_txt) in enumerate(zip(output_text_clean, vllm_output_text)):
        #     if decode_txt != vllm_txt:
        #         print(f"i={i}, decode_txt={decode_txt}, vllm_txt={vllm_txt}")
        
        num_gen_tokens = output.non_tensor_batch['gen_response_lengths'].tolist()
        stop_reasons = output.non_tensor_batch['stop_reasons'].tolist()
        # print(f"      Num gen tokens: {num_gen_tokens}")
        # print(f"      Stop reasons: {stop_reasons}")
        # print(f"      Returning {len(vllm_output_text)} outputs")

        # return output_text_clean, num_gen_tokens, stop_reasons, resp_lens
        return vllm_output_text, num_gen_tokens, stop_reasons, resp_lens

    def _update_history_and_check_finish(
        self,
        role: str,
        current_outputs: List[str],
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        finish_flag: str,
        agent_roles: List[str],
        num_gen_tokens: List[int],
        stop_reasons: List[Optional[str]],
        fact_prompts: List[str],
        executor_prompts: List[str],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        system_prompts: Dict[str, str],
        tokenizer: PreTrainedTokenizer,
    ):
        """Update conversation history and check completion flags"""
        # print(f"    _update_history_and_check_finish called for role={role}")
        # print(f"      Current outputs count: {len(current_outputs)}")
        # print(f"      Unfinished indices: {unfinished_indices.tolist() if hasattr(unfinished_indices, 'tolist') else unfinished_indices}")
        # print(f"      Finish flag: '{finish_flag}'")
        
        # Update history
        assert len(current_outputs) == len(unfinished_indices), \
            f'{len(current_outputs)} != {len(unfinished_indices)}'
        for i, idx in enumerate(unfinished_indices):
            history[idx].append({"role": role, "content": current_outputs[i], 
                                 "num_gen_tokens": num_gen_tokens[i], 
                                 "stop_reason": stop_reasons[i]})
        # print(f"      History updated for all unfinished samples")

        # Update finish flags
        # Check completion flags
        # print(f"      Checking finish flags...")
        if role == agent_roles[1]:
            # print(f"        Role is {agent_roles[1]}, checking for finish flag in previous outputs")
            for i, idx in enumerate(unfinished_indices):
                last_output = history[idx][-2]
                assert last_output["role"] == agent_roles[0]
                response = last_output['content']
                if finish_flag and finish_flag in response:
                    # print(f"          Sample {idx}: Found finish flag, marking as finished")
                    finish_flags[idx] = True
                    finish_reason[idx] = None
        
        if self.config.stop_when_truncated:
            # print(f"        Checking for truncation...")
            for i, stop_reason in enumerate(stop_reasons):
                # if stop_reason == "length" and not finish_flags[unfinished_indices[i]]:
                # XXX: even if stop by finish_flag, if current output is truncated, we need
                #  mark this trajectory as terminated
                if stop_reason == "length":
                    idx = unfinished_indices[i]
                    # print(f'idx={idx}, stop_when_truncated')
                    finish_flags[idx] = True
                    finish_reason[idx] = "stop_when_truncated"
                    if role == agent_roles[0]:
                        # update conversation for reasoning agent
                        # Use executor_prompts for reasoning agent, not fact_prompts
                        _, new_conversation = self._prepare_role_prompts(
                            agent_roles[1],
                            [idx],
                            history,
                            executor_prompts,
                            agent_roles,
                            system_prompts,
                            tokenizer,
                            conversation_history,
                        )
                        # Append empty assistant response to match conversation structure
                        new_conversation[0].append({"role": "assistant", "content": ""})
                        conversation_history[agent_roles[1]][idx] = new_conversation[0]

                        # add dummy history of reasoning agent
                        history[idx].append(
                            {"role": agent_roles[1], "content": "", 
                             "num_gen_tokens": 0, 
                             "stop_reason": "stop_when_truncated"}
                        )
        
        # print(f"      Finish checking completed")
        # print(f"      Total finished samples now: {finish_flags.sum()}")

    def _mark_unfinished_as_max_turns(self, finish_flags: np.ndarray,
                                      finish_reason: List[Optional[str]]):
        """Mark unfinished samples as reaching maximum turns"""
        # print(f"\n_mark_unfinished_as_max_turns called")
        unfinished_count = (~finish_flags).sum()
        # print(f"  Marking {unfinished_count} unfinished samples as 'reach_max_turn'")
        
        for i in range(len(finish_flags)):
            if not finish_flags[i]:
                finish_reason[i] = "reach_max_turn"
        
        # print(f"  All samples now have finish reasons")

    def _prepare_final_output(
        self,
        tensor_dict: Dict[str, Dict[str, torch.Tensor]],
        latest_outputs: List[str],
        history: List[List[Dict[str, str]]],
        finish_reason: List[Optional[str]],
        agent_roles: List[str],
        prompts: DataProto,
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        mem_op_stats: Optional[Dict[str, List[int]]] = None,
    ):
        """Prepare final output"""
        # print(f"\n_prepare_final_output called")
        # print(f"  Latest outputs count: {len(latest_outputs)}")
        # print(f"  History count: {len(history)}")
        # print(f"  Agent roles: {agent_roles}")

        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch["finish_reason"] = finish_reason
        non_tensor_batch["num_turns"] = [
            len(h) // len(agent_roles) for h in history
        ]
        non_tensor_batch["response"] = latest_outputs
        
        # Add memory operation statistics if available
        if mem_op_stats is not None:
            # Convert lists to numpy arrays with dtype=object for DataProto compatibility
            non_tensor_batch["mem_insert_successful"] = np.array(mem_op_stats['insert_successful'], dtype=object)
            non_tensor_batch["mem_delete_successful"] = np.array(mem_op_stats['delete_successful'], dtype=object)
            non_tensor_batch["mem_update_successful"] = np.array(mem_op_stats['update_successful'], dtype=object)
            non_tensor_batch["mem_insert_total"] = np.array(mem_op_stats['insert_total'], dtype=object)
            non_tensor_batch["mem_delete_total"] = np.array(mem_op_stats['delete_total'], dtype=object)
            non_tensor_batch["mem_update_total"] = np.array(mem_op_stats['update_total'], dtype=object)
            # Add dia_ids affected per turn for turn-level causal reward assignment
            # Store as list of lists directly (each sample has a list of turn operations)
            # Create as 1D object array to avoid concatenation issues across batches
            dia_ids_array = np.empty(len(mem_op_stats['dia_ids_affected_per_turn']), dtype=object)
            for i, dia_ids_list in enumerate(mem_op_stats['dia_ids_affected_per_turn']):
                dia_ids_array[i] = dia_ids_list
            non_tensor_batch["dia_ids_affected_per_turn"] = dia_ids_array            
        # print(f"  Non-tensor batch updated with finish_reason, num_turns, response")

        padded_history = _pad_history(history, 2 * self.config.max_num_turns)
        padded_conversation_history = {
            role:
            # 1 + 2 * self.config.max_num_turns to account for system prompt
            _pad_history(conversation_history[role],
                         1 + 2 * self.config.max_num_turns)
            for role in agent_roles
        }

        non_tensor_batch["history"] = padded_history
        for role in agent_roles:
            non_tensor_batch[
                f"{role}_conversation_history"] = padded_conversation_history[
                    role]

        flat_tensor_dict = {}
        for role in tensor_dict.keys():
            for key in tensor_dict[role].keys():
                flat_tensor_dict[f"{role}_{key}"] = tensor_dict[role][key]
        
        # print(f"  Flattened tensor dict with {len(flat_tensor_dict)} keys: {list(flat_tensor_dict.keys())}")
        # print(f"  Creating final DataProto...")

        final_output = DataProto.from_dict(
            tensors=flat_tensor_dict,
            non_tensors=non_tensor_batch,
            meta_info=prompts.meta_info,
        )
        # print(f"  Final output prepared successfully")
        return final_output


def _pad_history(input_historys: List[List[Dict[str, str]]],
                 max_length: int,
                 pad_value={
                     "role": "padding",
                     "content": "<PAD>"
                 }):
    padded_history = []
    for history in input_historys:
        current_length = len(history)
        pad_length = max_length - current_length
        assert pad_length >= 0, f"current_length: {current_length}, max_length: {max_length}"
        padded_history.append(history + [pad_value] * pad_length)
    return padded_history

def encode_conversation(conversation: List[Dict[str, str]], 
                        tokenizer: PreTrainedTokenizer, 
                        num_gen_tokens: List[int], 
                        stop_reasons: List[Optional[str]]):
    IGNORE_INDEX = -100
    input_ids = []
    labels = [] 
    step_ids = []
    cur_len = 0
    cur_hist = []
    i_step = 0
    for i, msg in enumerate(conversation):
        if msg['role'] in ['system', 'user']:
            pass
        elif msg['role'] == 'assistant':
            # query string
            query = tokenizer.apply_chat_template(
                cur_hist,
                add_generation_prompt=True,
                tokenize=False
            )
            # response string
            response = msg['content']
            query_ids = tokenizer.encode(query, add_special_tokens=True)
            query_response_ids = tokenizer.encode(query + response, add_special_tokens=True)
            response_ids = query_response_ids[len(query_ids):]
            input_ids = query_response_ids

            ################################################################
            # input_ids: 
            # | this | is | a | test | <im_end> | <im_start> | <assistant> | this | is | a | response | <im_end> |
            # query_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> |
            # response_ids:
            # | this | is | a | response | <im_end> |
            # step_ids:
            # |IGNORE| IG |IG | IG   | IG       | IG         | i_step      |i_step| ... |i_step| IGNORE |
            # labels:
            # |IGNORE| IG |IG | IG   | IG       | IG         | this | is   | a | response   | <im_end> | IGNORE
            #################################################################
            step_ids.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))
            labels.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))

            stop_reason = stop_reasons[i_step]
            # if stop normally, add eos token
            if stop_reason == "stop":
                labels.extend(response_ids + [tokenizer.eos_token_id])
                step_ids.extend([i_step] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = len(response_ids) + 1
            # if truncated, do not add eos token as label
            elif stop_reason == "length":
                # print("# STOP REASON:", stop_reasons[i_step])
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([i_step] * len(response_ids) + [IGNORE_INDEX])
                num_gen_tokens[i_step] = len(response_ids)
            elif stop_reason in ['stop_when_truncated', 'completion_token_exceeded']:
                # special case for dummy response
                # XXX: in this case, response == ""
                assert response == ""
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([IGNORE_INDEX] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = 0
                break

            i_step += 1
            cur_len = len(query_response_ids)
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")
        cur_hist.append(msg)

    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
    return input_ids, labels, step_ids