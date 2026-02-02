import numpy as np
from omegaconf import DictConfig
from verl import DataProto
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import collate_fn as data_proto_collate_fn, pad_dataproto_to_divisor, unpad_dataproto
import torch
import unicodedata
import json
from verl.rema_trainer.memory.memory_core.memory import Memory
from verl.rema_trainer.memory.memory_core.memory_manager import MemoryManager
from verl.rema_trainer.memory.memory_core.prompt_generator import format_turns_for_prompt, generate_memory_prompt_using_facts
from verl.rema_trainer.memory.utils.parse_response import extract_llm_json_from_response

def normalize_text(text):
    return unicodedata.normalize('NFKC', text)

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


def _encode_conversation(
    conversation: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    num_gen_tokens: List[int],
    stop_reasons: List[Optional[str]],
):
    IGNORE_INDEX = -100
    labels = []
    step_ids = []
    cur_len = 0
    cur_hist = []
    i_step = 0
    for i, msg in enumerate(conversation):
        if msg["role"] in ["system", "user"]:
            pass
        elif msg["role"] == "assistant":
            # query string
            query = tokenizer.apply_chat_template(cur_hist,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
            # response string
            response = msg["content"]
            query_ids = tokenizer.encode(query, add_special_tokens=True)
            query_response_ids = tokenizer.encode(query + response,
                                                  add_special_tokens=True)
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
            elif stop_reason in [
                    "stop_when_truncated", "completion_token_exceeded"
            ]:
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


class MultiAgentRollout:

    def __init__(
        self, 
        config: DictConfig,
        tokenizers: Dict[str, PreTrainedTokenizer],
        rollout_wg_dict: Dict[str, RayWorkerGroup]
    ):
        self.config = config
        self.tokenizers = tokenizers
        self.rollout_wg_dict = rollout_wg_dict

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

            prompt = "{turns}"
            formatted_turns = format_turns_for_prompt(turns_data)
            prompt = prompt.format(turns=formatted_turns)
            prompts.append(prompt)

        return prompts
    
    def generate_memory_prompts(self, sample_ids, chunk_ids, facts_responses, epochs, split, conv_memories=None) -> Tuple[List[str], List[Memory], MemoryManager]:
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
                
                # Load memory from previous chunk if it exists
                if chunk_id > 1:  # chunk_id starts from 1
                    prev_chunk_id = chunk_id - 1
                    # Load the memory snapshot that the previous chunk_id had saved as our starting memory state
                    loaded_memory = shared_manager.get_snapshot(conv_id, prev_chunk_id, epoch, split)
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

    def _apply_chat_template(self, chat_lst: List[List[Dict[str, str]]],
                             tokenizer: PreTrainedTokenizer):
        """Apply chat template and encode"""
        return tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def _initialize_conversation_state(self, batch_size):
        """Initialize conversation state variables"""
        history = [[] for _ in range(batch_size)]
        finish_flags = np.zeros(batch_size, dtype=bool)
        finish_reason = [None for _ in range(batch_size)]
        return history, finish_flags, finish_reason

    def _build_chat_list_for_role(
        self,
        role: str,
        history_list: List[List[Dict[str, str]]],
        questions: List[str],
        system_prompts: Dict[str, str],
        agent_roles: List[str],
        existing_conversations: List[List[Dict[str, str]]] = None,
    ):
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
                            "You can refer to them for disambiguation or clarity, but do NOT treat them as new events. "
                            "Analyze the new conversation turns below and generate the new facts.\n```"
                            + question + "```"
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

        return chat_lst

    def _prepare_role_prompts(
        self,
        role: str,
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        questions: List[str],
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
    ) -> Tuple[DataProto, List[List[Dict[str, str]]]]:
        """Prepare prompts for a specific role"""

        # Prepare history and questions for currently unfinished samples
        current_history = [history[idx] for idx in unfinished_indices]
        current_questions = [questions[idx] for idx in unfinished_indices]

        # Build chat list
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

        # Apply chat template and encode
        inputs = self._apply_chat_template(chat_lst, tokenizers[role])
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        return data, chat_lst

    def _filter_truncated_prompts_before_generation(
        self,
        prompt_proto: DataProto,
        chat_lst: List[List[Dict[str, str]]],
        role: str,
        agent_roles: List[str],
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizer: PreTrainedTokenizer,
        unfinished_indices: np.ndarray,
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        i_turn: int,
    ):
        # check current state length
        non_trunc_input = tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=False,
            max_length=None,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        # state length
        seq_lens = non_trunc_input["attention_mask"].sum(dim=1).tolist()
        # if state length is larger than prompt length, the trajectory is terminated
        if not all([l <= self.config.prompt_length for l in seq_lens]):
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
                    print(f"idx={idx}, completion_token_exceeded")
                    # if the next gen is for reasoning agent, we need to add a dummy response in history
                    if role == agent_roles[1]:
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "completion_token_exceeded",
                        })
                        # update conversation history for reasoning agent
                        conversation_history[agent_roles[1]][idx] = chat_lst[i]
                    else:
                        if i_turn == 0:
                            raise RuntimeError(
                                f"1st round prompt larger than prompt length: {seq_lens[i]} > {self.config.prompt_length}"
                            )

            if len(new_prompt_protos):
                # collate prompt needed to generate this round
                new_prompt_proto = data_proto_collate_fn(new_prompt_protos)
                new_prompt_proto.meta_info = prompt_proto.meta_info
            else:
                new_prompt_proto = None
            return new_prompt_proto, new_chat_lst, new_unfinished_indices
        else:
            return prompt_proto, chat_lst, unfinished_indices

    def _generate_role_responses(
        self,
        rollout: RayWorkerGroup,
        prompt_proto: DataProto,
        tokenizer: PreTrainedTokenizer,
        response_length: int,
    ):
        """Generate responses for the current role"""
        pad_prompt_proto, pad_size = pad_dataproto_to_divisor(prompt_proto, rollout.world_size)
        output = rollout.raw_generate_sequences(pad_prompt_proto)
        unpad_output = unpad_dataproto(output, pad_size=pad_size)
        resp_lens = (unpad_output.batch["attention_mask"][:, -response_length:].sum(
            dim=1).tolist())
        vllm_output_text = unpad_output.non_tensor_batch["text"].tolist()

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

        num_gen_tokens = unpad_output.non_tensor_batch[
            "gen_response_lengths"].tolist()
        stop_reasons = unpad_output.non_tensor_batch["stop_reasons"].tolist()

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
        questions: List[str], # XXX: this is kept but we will use fact_prompts/executor_prompts
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        system_prompts: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
        fact_prompts: List[str] = None, # Added
        executor_prompts: List[str] = None, # Added
    ):
        """Update conversation history and check completion flags"""
        # Update history
        assert len(current_outputs) == len(
            unfinished_indices
        ), f"{len(current_outputs)} != {len(unfinished_indices)}"
        for i, idx in enumerate(unfinished_indices):
            history[idx].append({
                "role": role,
                "content": current_outputs[i],
                "num_gen_tokens": num_gen_tokens[i],
                "stop_reason": stop_reasons[i],
            })

        # Update finish flags
        # Check completion flags
        if role == agent_roles[1]:
            for i, idx in enumerate(unfinished_indices):
                last_output = history[idx][-2]
                assert last_output["role"] == agent_roles[0]
                response = last_output["content"]
                if finish_flag and finish_flag in response:
                    finish_flags[idx] = True
                    finish_reason[idx] = None

        if self.config.stop_when_truncated:
            for i, stop_reason in enumerate(stop_reasons):
                # if stop_reason == "length" and not finish_flags[unfinished_indices[i]]:
                # XXX: even if stop by finish_flag, if current output is truncated, we need
                #  mark this trajectory as terminated
                if stop_reason == "length":
                    idx = unfinished_indices[i]
                    print(f"idx={idx}, stop_when_truncated")
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
                            tokenizers,
                            conversation_history,
                        )
                        # Append empty assistant response to match conversation structure
                        new_conversation[0].append({"role": "assistant", "content": ""})
                        conversation_history[
                            agent_roles[1]][idx] = new_conversation[0]

                        # add dummy history of reasoning agent
                        history[idx].append({
                            "role":
                            agent_roles[1],
                            "content":
                            "",
                            "num_gen_tokens":
                            0,
                            "stop_reason":
                            "stop_when_truncated",
                        })

    def _run_multi_turn_conversation(
        self,
        prompts: DataProto,
        tokenizers: Dict[str, PreTrainedTokenizer],
        max_num_turns: int,
        agent_roles: List[str],
        system_prompts: Dict[str, str],
        finish_flag: str,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
    ):
        print(f"\n_run_multi_turn_conversation called")
        print(f"len of prompts.non_tensor_batch['turns_json']: {len(prompts.non_tensor_batch['turns_json'])}")

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
        }

        questions = prompts.non_tensor_batch["question"]
        assert len(finish_flags) == len(
            questions), f"{finish_flags.shape} != {len(questions)}"

        conversation_history = {
            role: [None for _ in range(len(questions))]
            for role in agent_roles
        }

        for i_turn in range(max_num_turns):
            # Get indices of unfinished samples
            unfinished_indices = np.where(~finish_flags)[0]
            print(f"turn {i_turn+1} of {max_num_turns}, \
                    {len(unfinished_indices)}/{len(questions)} unfinished")

            if len(unfinished_indices) == 0:
                break
            
            # Regenerate fact prompts for the current turn chunk
            fact_prompts = self.generate_fact_prompts(
                prompts.non_tensor_batch["turns_json"], 
                current_turn=i_turn,  # Use current turn index to get correct chunk
                max_turns=max_num_turns
            )
            fact_prompts = np.array(fact_prompts, dtype=object)

            # Each role takes turns generating in every round
            for i_role, role in enumerate(agent_roles):
                print(f"role: {role}")

                # Choose questions based on role
                if role == agent_roles[0]:
                    current_questions = fact_prompts
                else:
                    current_questions = executor_prompts

                # Prepare prompts for current role
                prompt_proto, chat_lst = self._prepare_role_prompts(
                    role,
                    unfinished_indices,
                    history,
                    current_questions,
                    agent_roles,
                    system_prompts,
                    tokenizers,
                    conversation_history,
                )

                # side effect on convsersation_history and history
                prompt_proto, chat_lst, unfinished_indices = (
                    self._filter_truncated_prompts_before_generation(
                        prompt_proto=prompt_proto,
                        chat_lst=chat_lst,
                        role=role,
                        agent_roles=agent_roles,
                        history=history,
                        conversation_history=conversation_history,
                        tokenizer=tokenizers[role],
                        unfinished_indices=unfinished_indices,
                        finish_flags=finish_flags,
                        finish_reason=finish_reason,
                        i_turn=i_turn,
                    ))
                if len(unfinished_indices) == 0:
                    break

                prompt_proto.meta_info.update(prompts.meta_info)
                # for i, chat in enumerate(chat_lst):
                #     idx = unfinished_indices[i]
                #     conversation_history[role][idx] = chat

                # Generate responses for current role
                current_outputs, num_gen_tokens, stop_reasons, resp_lens = (
                    self._generate_role_responses(
                        rollout=self.rollout_wg_dict[role],
                        prompt_proto=prompt_proto,
                        tokenizer=tokenizers[role],
                        response_length=response_length,
                    ))
                
                if role == agent_roles[0]:
                    # Update executor questions for next role
                    print(f"  Updating executor questions for next role...")
                    extracted_facts = [current_outputs[i] for i in range(len(current_outputs))]
                    # print(f"  Extracted facts for {len(extracted_facts)} samples, \nfacts:: {extracted_facts}")
                    
                    # Filter ALL inputs to unfinished samples
                    unfinished_sample_ids = [prompts.non_tensor_batch["sample_id"][idx] for idx in unfinished_indices]
                    unfinished_chunk_ids = [prompts.non_tensor_batch["chunk_id"][idx] for idx in unfinished_indices]
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
                    print(f"  Executing memory operations for {len(current_outputs)} samples...")
                    rewards, ops_per_sample, batch_op_stats = self._execute_memory_operations(
                        prompts, conv_memories, shared_manager, current_outputs, unfinished_indices
                    )
                    # Accumulate operation statistics for unfinished samples
                    for i, idx in enumerate(unfinished_indices):
                        for key in mem_op_stats.keys():
                            mem_op_stats[key][idx] += batch_op_stats[key][i]

                # XXX(ziyu): remove finish flag in output for reasoning agent here
                #  consider move to a post-processing function
                if role == agent_roles[1] and finish_flag:
                    current_outputs = [
                        output.replace(finish_flag, "").rstrip()
                        for output in current_outputs
                    ]
                
                # Append assistant responses to chat_lst and store complete conversation
                for i, output in enumerate(current_outputs):
                    chat_lst[i].append({"role": "assistant", "content": output})
                    idx = unfinished_indices[i]
                    conversation_history[role][idx] = chat_lst[i]

                # XXX(ziyu): side effect on `history`
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
                    fact_prompts, # passed fact_prompts
                    executor_prompts, # passed executor_prompts
                    conversation_history,
                    system_prompts,
                    tokenizers,
                )
                unfinished_indices = np.where(~finish_flags)[0]
                if len(unfinished_indices) == 0:
                    break
        
        # Now for conv_memories we need to save their snapshots after all turns
        # To avoid overwriting, will save with index in the batch
        # Use rollout indices computed AFTER repeating (not batch_idx which is set before repeating)
        try:
            rollout_batch_indices = prompts.batch['rollout_idx'].cpu().numpy().tolist()
        except KeyError:
            # Fallback if rollout_idx is not available (e.g. initial validation)
            rollout_batch_indices = list(range(len(prompts.non_tensor_batch["sample_id"])))

        sample_ids = prompts.non_tensor_batch["sample_id"]
        
        if conv_memories is not None:
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
                    index_in_batch=global_idx
                )
                # print(f"Saved memory snapshot for conv {conv_id}, chunk {chunk_id} at global index {global_idx}")

        # use the last output of each agent as latest output response
        latest_outputs = [h[-1]["content"] for h in history]
        return latest_outputs, conversation_history, conv_memories, shared_manager, mem_op_stats

    def _execute_memory_operations(
        self,
        prompt_data: DataProto,
        memories: List[Memory],
        shared_manager: MemoryManager,
        response_texts,
        unfinished_indices: List[int] = None,
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
        }
        sample_ids = prompt_data.non_tensor_batch["sample_id"]
        chunk_ids = prompt_data.non_tensor_batch["chunk_id"]
        turns_json = prompt_data.non_tensor_batch["turns_json"]
        
        # If unfinished_indices not provided, assume all samples
        if unfinished_indices is None:
            unfinished_indices = list(range(len(memories)))
        
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
            # Check range
            if response_idx >= len(response_texts):
                print(f"Error: response_idx {response_idx} out of range for response_texts length {len(response_texts)}")
                rewards.append(0.0)
                continue

            response_text = response_texts[response_idx]
            # Parse operations from response
            response_json = extract_llm_json_from_response(response_text)
            json_parse_success = response_json.get("_parse_success", False)
            operations = response_json.get("operations", [])
            
            # If JSON parsing failed, reward is 0
            if not json_parse_success:
                rewards.append(0.0)
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
            
            # Calculate operation success rate
            if total_ops == 0:
                # No operations - intentional, so 100% success
                ops_reward = 1.0
            else:
                ops_reward = successful_ops / total_ops
            
            # Final reward: JSON correct (1.0) * operation success rate
            final_reward = 1.0 * ops_reward
            rewards.append(final_reward)
            
            if result["status"] not in ["success", "partial"]:
                print(f"Warning: Memory operations had issues: {result}")
        
        return rewards, memory_operations_per_sample, operation_stats

    def _mark_unfinished_as_max_turns(self, finish_flags: np.ndarray,
                                      finish_reason: List[Optional[str]]):
        """Mark unfinished samples as reaching maximum turns"""
        for i in range(len(finish_flags)):
            if not finish_flags[i]:
                finish_reason[i] = "reach_max_turn"

    def _build_tensor_dict(
        self,
        last_round_responses: List[Dict[str, str]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizers: Dict[str, PreTrainedTokenizer],
        num_gen_token_lst: Dict[str, List[List[int]]],
        stop_reason_lst: Dict[str, List[List[Optional[str]]]],
        max_num_turns: int,
        finish_reason: List[Optional[str]],
    ):
        # conversation_history already contains full conversations with assistant responses
        # add last round output to make full conversation
        # for i_batch in range(len(last_round_responses)):
        #     for role in last_round_responses[i_batch]:
        #         conversation_history[role][i_batch].append({
        #             "role":
        #             "assistant",
        #             "content":
        #             last_round_responses[i_batch][role],
        #         })

        input_ids_lst = {role: [] for role in conversation_history.keys()}
        labels_lst = {role: [] for role in conversation_history.keys()}
        step_ids_lst = {role: [] for role in conversation_history.keys()}

        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                # encode conversation into input_ids, labels, step_ids
                # XXX(ziyu): need to consider stop reason here ?
                input_ids, labels, step_ids = _encode_conversation(
                    conversation_history[role][i_batch],
                    tokenizers[role],
                    num_gen_token_lst[role][i_batch],
                    stop_reason_lst[role][i_batch],
                )
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
            if max_length > self.config.response_length + self.config.prompt_length:
                print(
                    f"role: {role}, max_length={max_length} > {self.config.response_length + self.config.prompt_length}"
                )
                # raise RuntimeError(f"max_length={max_length} > {self.config.response_length + self.config.prompt_length}")

            # Use max length for padding and gathering
            max_length = self.config.response_length + self.config.prompt_length

            # Pad and convert to tensors
            padded_input_ids = torch.full((batch_size, max_length),
                                          tokenizers[role].pad_token_id,
                                          dtype=torch.long)
            padded_labels = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            padded_step_ids = torch.full(
                (batch_size, max_length),
                -100,
                dtype=torch.long  # IGNORE_INDEX
            )
            attention_mask = torch.zeros((batch_size, max_length),
                                         dtype=torch.long)

            # Fill in the actual values
            for i, (input_ids, labels, step_ids) in enumerate(
                    zip(input_ids_lst[role], labels_lst[role],
                        step_ids_lst[role])):
                seq_len = min(len(input_ids), max_length)
                padded_input_ids[i, :seq_len] = torch.tensor(
                    input_ids[:seq_len], dtype=torch.long)
                padded_labels[i, :seq_len] = torch.tensor(labels[:seq_len],
                                                          dtype=torch.long)
                padded_step_ids[i, :seq_len] = torch.tensor(step_ids[:seq_len],
                                                            dtype=torch.long)
                attention_mask[i, :seq_len] = 1

            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)

            padded_num_gen_tokens = torch.full((batch_size, max_num_turns),
                                               0,
                                               dtype=torch.long)
            for i, num_gen_tokens in enumerate(num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(
                    num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns),
                                             0,
                                             dtype=torch.bool)

            for i, stop_reasons in enumerate(stop_reason_lst[role]):
                stop_reason_array = np.array(
                    [0 if r == "stop" else 1 for r in stop_reasons])
                padded_stop_reasons[i, :len(stop_reason_array)] = torch.tensor(
                    stop_reason_array, dtype=torch.bool)

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
                }, )

        # # remove side effect
        # for i_batch in range(len(last_round_responses)):
        #     for role in last_round_responses[i_batch]:
        #         conversation_history[role][i_batch].pop()

        return tensor_dict

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
            # print(f"  Added memory operation statistics to non_tensor_batch (batch_size={len(mem_op_stats['insert_successful'])})")

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

        return DataProto.from_dict(
            tensors=flat_tensor_dict,
            non_tensors=non_tensor_batch,
            meta_info=prompts.meta_info,
        )
    
    def _checking(
        self,
        history: List[List[Dict[str, str]]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        agent_roles: List[str],
        last_round_responses: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        tensor_dict: Dict[str, torch.tensor],
        final_output: DataProto,
    ):
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

                assert normalize_text(query_response) == normalize_text(raw_query_response), \
                    f"'{query_response}' != '{raw_query_response}'"
                for i_turn in range(num_turn):
                    turn_labels = labels[step_ids == i_turn]
                    if stop_reasons[i_turn] == 0:
                        assert turn_labels[-1] == tokenizer.eos_token_id
                        turn_labels = turn_labels[:-1] # drop eos
                    response = tokenizer.decode(turn_labels.tolist())
                    assert normalize_text(response) == normalize_text(history[i][i_role + i_turn * len(agent_roles)]['content']), \
                        f"'{response}' != '{history[i][i_role + i_turn * len(agent_roles)]['content']}'"
        

    def generate(
        self,
        prompts: DataProto,
        **kwargs,
    ) -> DataProto:
        """Main function responsible for coordinating multi-turn dialogue generation"""
        
        # Extract meta info
        max_num_turns = prompts.meta_info['max_num_turns']
        agent_roles = prompts.meta_info['agent_roles']
        finish_flag = prompts.meta_info['finish_flag']
        system_prompts = prompts.meta_info['system_prompts']
        
        rollout_wg = self.rollout_wg_dict

        # tokenizers = {role: wg.tokenizer for role, wg in rollout_wg.items()}
        tokenizers = self.tokenizers
        for role in rollout_wg.keys():
            tokenizers[role].padding_side = "left"
            if tokenizers[role].pad_token is None:
                tokenizers[role].pad_token = tokenizers[role].eos_token

        prompts.meta_info['is_multi_turn'] = True

        sample_ids = prompts.non_tensor_batch["sample_id"]
        batch_size = len(sample_ids)
        
        # Initialize state variables
        history, finish_flags, finish_reason = self._initialize_conversation_state(
            batch_size)

        # Multi-turn dialogue generation
        latest_outputs, conversation_history, conv_memories, shared_manager, mem_op_stats = self._run_multi_turn_conversation(
            prompts,
            tokenizers=self.tokenizers,
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

        # Mark completion reasons
        if max_num_turns > 1:
            self._mark_unfinished_as_max_turns(finish_flags, finish_reason)

        last_round_responses = [{
            m["role"]: m["content"]
            for m in h[-2:]
        } for h in history]

        # extract information from history record
        num_gen_token_lst = {role: [] for role in agent_roles}
        stop_reason_lst = {role: [] for role in agent_roles}
        for h in history:
            _num_gen_tokens = {role: [] for role in agent_roles}
            _stop_reasons = {role: [] for role in agent_roles}
            for m in h:
                _num_gen_tokens[m["role"]].append(m["num_gen_tokens"])
                _stop_reasons[m["role"]].append(m["stop_reason"])
            for role in agent_roles:
                num_gen_token_lst[role].append(_num_gen_tokens[role])
                stop_reason_lst[role].append(_stop_reasons[role])

        tensor_dict = self._build_tensor_dict(
            last_round_responses,
            conversation_history,
            tokenizers,
            num_gen_token_lst,
            stop_reason_lst,
            max_num_turns,
            finish_reason,
        )

        final_output = self._prepare_final_output(
            tensor_dict=tensor_dict,
            latest_outputs=latest_outputs,
            history=history,
            finish_reason=finish_reason,
            agent_roles=agent_roles,
            prompts=prompts,
            conversation_history=conversation_history,
            mem_op_stats=mem_op_stats, # Added
        )

        if self.config.add_checking:
            try:
                self._checking(
                    history=history,
                    conversation_history=conversation_history,
                    agent_roles=agent_roles,
                    last_round_responses=last_round_responses,
                    tokenizer=tokenizers[agent_roles[0]],
                    tensor_dict=tensor_dict,
                    final_output=final_output,
                )
            except AssertionError as e:
                print("Error during checking:", e)
        
        return final_output
