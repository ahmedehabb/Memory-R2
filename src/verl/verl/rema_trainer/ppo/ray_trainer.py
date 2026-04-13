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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
from pathlib import Path
import uuid
import jsonlines
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.rema_trainer.ppo import core_algos
from verl.rema_trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.rema_trainer.memory.teacher_model import TeacherModel
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rema_dataset import RLHFDataset, collate_fn, ChunkBatchSampler
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import torch_functional as verl_F

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    GRPO_MAXRL = 'grpo_maxrl'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        import time
        import logging
        
        timeout = 300  # 300 seconds = 5 minutes
        retry_interval = 10  # seconds
        start_time = time.time()
        
        while True:
            node_available_resources = ray.state.available_resources_per_node()
            node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

            # check total required gpus can be satisfied
            total_available_gpus = sum(node_available_gpus.values())
            total_required_gpus = sum(
                [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
            
            # Check for resource pool satisfaction
            pools_satisfied = True
            error_msgs = []
            
            if total_available_gpus < total_required_gpus:
                pools_satisfied = False
                error_msgs.append(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
            else:
                # check each resource pool can be satisfied, O(#resource_pools * #nodes)
                for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
                    num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
                    for node, available_gpus in node_available_gpus.items():
                        if available_gpus >= num_gpus:
                            node_available_gpus[node] -= num_gpus
                            num_nodes -= 1
                            if num_nodes == 0:
                                break
                    if num_nodes > 0:
                        pools_satisfied = False
                        error_msgs.append(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster")
            
            # If all resources are available, return
            if pools_satisfied:
                return
            
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                # If we've timed out, raise the error with all collected error messages
                raise ValueError(f"Resource allocation timed out after {timeout} seconds. Errors: {'; '.join(error_msgs)}")
            
            # Log waiting message and sleep before retry
            remaining = timeout - elapsed_time
            logging.info(f"Waiting for resources to be available. Retrying in {retry_interval} seconds. Timeout in {remaining:.1f} seconds.")
            logging.info(f"Resource issues: {'; '.join(error_msgs)}")
            time.sleep(retry_interval)


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty='kl'):
    """Apply KL penalty to token-level rewards using step_ids-based masking.
    
    Computes KL divergence between old and reference log probs, masks it
    with valid step positions (step_ids != -100), and subtracts the
    weighted KL from token_level_scores to produce token_level_rewards.
    Also updates the adaptive KL controller.
    """
    step_mask = (data.batch['step_ids'] != -100).float()
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)
    kld = kld * step_mask
    beta = kl_ctrl.value
    data.batch['token_level_rewards'] = data.batch['token_level_scores'] - beta * kld

    current_kl = masked_mean(kld, mask=step_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=len(data.batch))

    metrics = {'critic/kl_in_reward': current_kl, 'critic/kl_coeff': beta}
    return data, metrics


def compute_bootstrap_values(data: DataProto, gamma_session: float = 1.0) -> torch.Tensor:
    """Compute bootstrap values for cross-session value propagation.
    
    For each row in the batch, finds the next session in the same 
    trajectory chain and extracts the first turn's critic value 
    from that next session. This enables the GAE computation to bootstrap
    from the next session's value instead of treating session boundaries as terminal.
    
    After merge_roles_data, the batch has 2x rows (role-A rows then role-B rows).
    We use uid prefix (role name) + sample_id + rollout_idx to build chains,
    ensuring each role's trajectory is separate.
    
    Args:
        data: DataProto with 'values', 'step_ids', 'rollout_idx' in batch,
              and 'uid', 'sample_id', 'session_id' in non_tensor_batch.
        gamma_session: Cross-session discount factor. Bootstrap values are
                       multiplied by gamma_session to dampen future session signal.
    
    Returns:
        bootstrap_values: (bs,) tensor. Zero for terminal sessions (last in chain).
    """
    values = data.batch['values']
    step_ids = data.batch['step_ids']
    rollout_idxs = data.batch['rollout_idx']
    if isinstance(rollout_idxs, torch.Tensor):
        rollout_idxs = rollout_idxs.cpu().tolist()
    
    sample_ids = data.non_tensor_batch.get('sample_id', None)
    session_ids = data.non_tensor_batch.get('session_id', None)
    uids = data.non_tensor_batch.get('uid', None)
    
    bs = values.shape[0]
    bootstrap_values = torch.zeros(bs, device=values.device, dtype=values.dtype)
    
    if sample_ids is None or session_ids is None:
        return bootstrap_values
    
    # Determine role prefix for each row to separate role chains after merge_roles_data
    # After merge, uid format is "rolename_<uuid>" where rolename is from agent_roles
    agent_roles = data.meta_info.get('agent_roles', [])
    
    from collections import defaultdict
    chain_map = defaultdict(list)
    for row_idx in range(bs):
        # Extract role prefix from uid
        role_prefix = ""
        if uids is not None:
            uid_str = str(uids[row_idx])
            for role in agent_roles:
                if uid_str.startswith(f"{role}_"):
                    role_prefix = role
                    break
        
        # Chain key: (role, sample_id, rollout_idx)
        chain_key = (role_prefix, str(sample_ids[row_idx]), int(rollout_idxs[row_idx]))
        sess_id = int(session_ids[row_idx])
        chain_map[chain_key].append((sess_id, row_idx))
    
    # Sort each chain by session_id and link consecutive sessions
    n_chains = len(chain_map)
    chain_lengths = []
    n_misses = 0  # Track how many links fail to find valid tokens
    for key, sessions in chain_map.items():
        sessions.sort(key=lambda x: x[0])  # Sort by session_id
        chain_lengths.append(len(sessions))
        for i in range(len(sessions) - 1):
            current_row_idx = sessions[i][1]
            next_row_idx = sessions[i + 1][1]
            
            # Extract the value at the first token of turn 0 of the next session
            # This represents V(s_boundary) = expected return BEFORE any next-session actions
            next_step_ids = step_ids[next_row_idx]  # (seq_len,)
            next_values = values[next_row_idx]  # (seq_len,)
            
            turn0_mask = (next_step_ids == 0)
            if turn0_mask.any():
                first_pos = turn0_mask.nonzero(as_tuple=True)[0][0]
                # Discount by gamma_session for the cross-session hop
                bootstrap_values[current_row_idx] = gamma_session * next_values[first_pos].detach()
            else:
                n_misses += 1
        # Last session in chain: bootstrap_values stays 0 (terminal)
    
    # Debug: show chain info
    from collections import Counter
    len_counts = Counter(chain_lengths)
    expected_bootstraps = sum(l - 1 for l in chain_lengths)
    print(f"[Bootstrap Debug] {n_chains} chains, length distribution: {dict(len_counts)}, "
          f"expected bootstraps: {expected_bootstraps}, misses: {n_misses}, "
          f"unique roles: {set(k[0] for k in chain_map.keys())}, "
          f"unique rollout_idxs: {set(k[2] for k in chain_map.keys())}")
    
    return bootstrap_values


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        step_ids = data.batch['step_ids']
        step_mask = (step_ids != -100).float()
        
        # Get bootstrap values for cross-session propagation
        bootstrap_value = data.batch.get('bootstrap_value', None)
        
        if data.meta_info.get('use_bilevel_gae', False):
            max_num_turns = data.meta_info['max_num_turns']
            # Use RAW per-turn rewards (not discounted turn_level_return)
            # because bi-level GAE handles cross-turn discounting internally
            bsz, seq_len = step_ids.shape
            token_level_rewards_raw = torch.zeros((bsz, seq_len), dtype=torch.float32, device=values.device)
            turn_level_reward = data.batch['turn_level_reward']
            for i_turn in range(max_num_turns):
                last_indices = get_last_index_of_turn(step_ids, i_turn)
                valid_mask = last_indices != -1
                if (~valid_mask).all():
                    break
                batch_indices = torch.arange(bsz, device=values.device)
                token_level_rewards_raw[batch_indices[valid_mask], last_indices[valid_mask]] = \
                    turn_level_reward[:, i_turn][valid_mask]
            # Also add any per-token KL penalty if it was applied
            if 'token_level_rewards' in data.batch and data.batch.get('token_level_scores', None) is not None:
                kl_penalty_per_token = data.batch['token_level_rewards'] - data.batch['token_level_scores']
                token_level_rewards_raw = token_level_rewards_raw + kl_penalty_per_token
            high_level_gamma = data.meta_info.get('gamma_turn_level', 1.0)
            advantages, returns = core_algos.compute_bi_level_gae_advantage_return(
                token_level_rewards=token_level_rewards_raw,
                values=values,
                eos_mask=step_mask,
                step_ids=step_ids,
                gamma=gamma,
                lam=lam,
                high_level_gamma=high_level_gamma,
                max_num_turns=max_num_turns,
                bootstrap_value=bootstrap_value,
            )
        else:
            token_level_rewards = data.batch['token_level_rewards']
            advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                          values=values,
                                                                          eos_mask=step_mask,
                                                                          gamma=gamma,
                                                                          lam=lam,
                                                                          bootstrap_value=bootstrap_value)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_sparse_rewards = torch.zeros_like(data.batch['token_level_rewards'])
        grpo_sparse_rewards[:, -1] = data.batch['turn_level_reward'].sum(-1)
        index = data.non_tensor_batch['uid']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=grpo_sparse_rewards,
                                                                        eos_mask=step_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_MAXRL:
        grpo_sparse_rewards = torch.zeros_like(data.batch['token_level_rewards'])
        grpo_sparse_rewards[:, -1] = data.batch['turn_level_reward'].sum(-1)
        index = data.non_tensor_batch['uid']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_grpo_maxrl_outcome_advantage(token_level_rewards=grpo_sparse_rewards,
                                                                        eos_mask=step_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=step_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        token_level_rewards = data.batch['token_level_rewards']
        grpo_sparse_rewards = torch.zeros_like(data.batch['token_level_rewards'])
        grpo_sparse_rewards[:, -1] = data.batch['turn_level_reward'].sum(-1)

        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        index = data.non_tensor_batch['uid']
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=grpo_sparse_rewards, eos_mask=step_mask, index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        raise NotImplementedError('REMAX is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        raise NotImplementedError('RLOO is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

def get_last_index_of_turn(step_ids: torch.Tensor, i_turn: int) -> torch.Tensor:
    mask = step_ids == i_turn
    seq_tensor = torch.arange(step_ids.size(1), device=step_ids.device).expand_as(step_ids)
    last_indices = torch.where(mask, seq_tensor, torch.tensor(-1, device=step_ids.device))
    last_indices, _ = torch.max(last_indices, dim=1)  # shape: [bsz]
    
    return last_indices

def compute_token_level_scores(data: DataProto, dtype=torch.float32)->torch.Tensor:
    max_num_turns = data.meta_info['max_num_turns']
    bsz, seq_len = data.batch['input_ids'].shape
    token_level_scores = torch.zeros((bsz, seq_len), dtype=torch.float32)
    step_ids = data.batch['step_ids']
    turn_level_return = data.batch['turn_level_return']
    for i_turn in range(max_num_turns):
        last_indices = get_last_index_of_turn(step_ids, i_turn)
        valid_mask = last_indices != -1
        if (~valid_mask).all(): break
        batch_indices = torch.arange(bsz)
        token_level_scores[batch_indices[valid_mask], last_indices[valid_mask]] = \
            turn_level_return[:, i_turn][valid_mask]
    
    return token_level_scores


def merge_roles_data(data: DataProto) -> DataProto:
    """
    Merge data from different roles into a single DataProto.
    
    Args:
        data: DataProto containing data for multiple roles with prefixed keys
        
    Returns:
        A new DataProto with merged data and transformed keys
    """
    agent_roles = data.meta_info['agent_roles']
    
    new_tensor_batch = {}
    for key in data.batch.keys():
        role_name = ''
        for role in agent_roles:
            if role in key:
                role_name = role
                break
        if role_name in agent_roles:
            v_name = key.replace(f'{role_name}_', '')
            if v_name not in new_tensor_batch:
                new_tensor_batch[v_name] = [None, None]
            new_tensor_batch[v_name][agent_roles.index(role_name)] = data.batch[key]
        else:
            new_tensor_batch[key] = data.batch[key].repeat(len(agent_roles), *[1 for _ in range(data.batch[key].ndim - 1)])
    new_tensor_batch['num_turns'] = torch.tensor(data.non_tensor_batch['num_turns'].tolist()).repeat(len(agent_roles))

    for key in new_tensor_batch.keys():
        if isinstance(new_tensor_batch[key], list):
            new_tensor_batch[key] = torch.cat(new_tensor_batch[key], dim=0)
    
    new_non_tensor_batch = {}
    uid_list = data.non_tensor_batch['uid'].tolist()
    new_uid_list = []
    for role in agent_roles:
        named_uid_list = [f'{role}_{uid}' for uid in uid_list]
        new_uid_list.extend(named_uid_list)
    new_non_tensor_batch['uid'] = np.array(new_uid_list, dtype=object)
    
    # Preserve session metadata for cross-session value bootstrapping
    for meta_key in ['sample_id', 'session_id']:
        if meta_key in data.non_tensor_batch:
            original_values = data.non_tensor_batch[meta_key].tolist()
            # Duplicate for each role (same session info applies to both roles)
            repeated_values = []
            for role in agent_roles:
                repeated_values.extend(original_values)
            new_non_tensor_batch[meta_key] = np.array(repeated_values, dtype=object)
    
    merged_data = DataProto.from_dict(new_tensor_batch, non_tensors=new_non_tensor_batch, meta_info=data.meta_info)
    return merged_data

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

class ReplayBuffer:
    """A tiny index-based replay buffer that stores dataset indices with provenance.

    - stores tuples of (dataset_index:int, orig_epoch:int)
    - supports uniform and simple recency-biased sampling
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self.buffer: list[tuple[int, int]] = []
        self.pos = 0

    def add_indices(self, indices, epoch: int = 0):
        """Add indices with the originating epoch provenance.

        Args:
            indices: iterable of ints
            epoch: int epoch to attach to each index
        """
        for idx in indices:
            entry = (int(idx), int(epoch))
            if len(self.buffer) < self.capacity:
                self.buffer.append(entry)
            else:
                self.buffer[self.pos] = entry
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, k: int, strategy: str = 'uniform'):
        import random

        if not self.buffer or k <= 0:
            return []
        k = min(k, len(self.buffer))
        if strategy == 'uniform':
            return random.sample(self.buffer, k)
        elif strategy == 'recency':
            # simple recency bias: sample from the most recent window
            window = self.buffer[-min(len(self.buffer), max(k * 4, 1)):]
            return random.sample(window, min(k, len(window)))
        else:
            return random.sample(self.buffer, k)

def merge_batch_dicts(dicts: list[dict]) -> dict:
    """Merge multiple collated batch dicts (tensors and non-tensors) along batch dim.

    Assumes all dicts have the same keys and that tensors should be concatenated on dim=0.
    """
    if not dicts:
        return {}
    out = {}
    keys = list(dicts[0].keys())
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        if not vals:
            continue
        first = vals[0]
        if isinstance(first, torch.Tensor):
            out[k] = torch.cat(vals, dim=0)
        elif isinstance(first, np.ndarray):
            out[k] = np.concatenate(vals, axis=0)
        elif isinstance(first, list):
            combined = []
            for v in vals:
                combined.extend(v)
            out[k] = combined
        else:
            # fallback: make numpy array of objects
            out[k] = np.concatenate([np.array(v, dtype=object) for v in vals], axis=0)
    return out


class RayReMATrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.GRPO_MAXRL, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        
        # since we will train two agents data
        self.config.actor_rollout_ref.actor.ppo_mini_batch_size *= 2

        self._create_dataloader()

        self.accumulated_batches = []

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"
        
        if config.algorithm.filter_groups.enable:
            assert config.actor_rollout_ref.rollout.n > 1
        
        if config.actor_rollout_ref.actor.clip_mode == 'turn':
            assert config.actor_rollout_ref.actor.agg_mode != 'token'
        
        if config.reward_model.get('use_format_reward', False):
            assert config.actor_rollout_ref.rollout.max_num_turns == 1, \
                "use_format_reward only support max_num_turns==1"


        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                        #  tokenizer=self.tokenizer,
                                        #  processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         shuffle=self.config.data.shuffle,
                                        #  image_key=self.config.data.get('image_key', 'images'),
                                        #  max_prompt_length=self.config.data.max_prompt_length,
                                        #  filter_prompts=True,
                                        #  return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        #  truncation=self.config.data.get('truncation', 'error'),
                                        #  filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                        )
        # TODO(ziyu): try to check data in dataset.
        #### UNUSED NOW
        # assert self.train_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
        
        # Use ChunkBatchSampler for training to ensure batches never span multiple chunk_ids
        # This is critical for memory management where chunk N depends on chunk N-1's saved memory
        # Note: ChunkBatchSampler doesn't support shuffle yet (processes chunks sequentially)
        if self.config.data.shuffle:
            raise NotImplementedError("shuffle=True is not supported with ChunkBatchSampler. Processing chunks sequentially.")
            print("WARNING: shuffle=True is not supported with ChunkBatchSampler. Processing chunks sequentially.")
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
            self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        else:
            print("INFO: Using ChunkBatchSampler for training dataloader to avoid cross-chunk memory issues.")
            train_batch_sampler = ChunkBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.config.data.train_batch_size,
                drop_last=False,  # Not needed since pad_incomplete=True always creates full batches
                pad_incomplete=False  # Pad training batches with repeats (acts like extra rollouts)
            )
        
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                # Use batch_sampler instead of batch_size + sampler
                batch_sampler=train_batch_sampler,
                num_workers=8,
                collate_fn=collate_fn
            )

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                    #    tokenizer=self.tokenizer,
                                    #    processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       shuffle=False,  # Validation should always be sequential
                                       #    image_key=self.config.data.get('image_key', 'images'),
                                       #    max_prompt_length=self.config.data.max_prompt_length,
                                       #    filter_prompts=True,
                                       #    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    #    truncation=self.config.data.get('truncation', 'error'),
                                    #    filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                       )
        # TODO(ziyu): try to check data in dataset.     
        ##### UNUSED NOW
        # assert self.val_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
                
        # ALL validation done with ChunkBatchSampler to avoid cross-chunk memory issues
        val_batch_sampler = ChunkBatchSampler(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            drop_last=False,  # Keep all samples
            pad_incomplete=False  # Don't pad validation batches - use actual sizes
        )
    
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Use batch_sampler instead of batch_size to control chunk-aware batching
            batch_sampler=val_batch_sampler,
            num_workers=8,
            collate_fn=collate_fn)

        # Create test dataset and dataloader if test_only, test_before_train, or test_after_train is enabled
        if self.config.trainer.get('test_only', False) or self.config.trainer.get('test_before_train', False) or self.config.trainer.get('test_after_train', True):
            print("INFO: test_only mode enabled, creating test dataset and dataloader")
            self.test_dataset = RLHFDataset(parquet_files=self.config.data.test_files,
                                           prompt_key=self.config.data.prompt_key,
                                           shuffle=False,  # Test should always be sequential
                                           )
            
            test_batch_sampler = ChunkBatchSampler(
                dataset=self.test_dataset,
                batch_size=self.config.data.get('test_batch_size', self.config.data.val_batch_size),
                drop_last=False,  # Keep all samples for test
                pad_incomplete=False  # Don't pad test batches - use actual sizes
            )
            
            self.test_dataloader = StatefulDataLoader(
                dataset=self.test_dataset,
                batch_sampler=test_batch_sampler,
                num_workers=8,
                collate_fn=collate_fn)
            
            print(f'Size of test dataloader: {len(self.test_dataloader)}')
        else:
            self.test_dataset = None
            self.test_dataloader = None

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        # With session accumulation, only ONE gradient update fires per epoch (at the final session).
        # Using len(dataloader)*epochs would be N_sessions× too large, causing is_last_step to
        # never trigger (final save/validation would be skipped).
        total_training_steps = self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores, groundtruths, histories):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, groundtruths, histories))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        # print("\n" + "="*80)
        # print("STARTING VALIDATION")
        # print("="*80)
        
        num_turns_lst = []
        history_lst = []
        sample_groundtruths = []
        completion_tokens_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # Lists to collect metrics
        reward_tensor_lst = []
        reward_tensor_dict_lst = []  # Store full dicts to access category metrics
        acc_tensor_lst = []
        bleu_tensor_lst = []
        data_source_lst = []

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        single_agent_mode = self.config.actor_rollout_ref.rollout.get('single_agent_mode', False)
        # print(f"\n[VALIDATE] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT, SINGLE_AGENT_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None, # changed this to None from FINISH_FLAG
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': SINGLE_AGENT_PROMPT if single_agent_mode else MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
        else:
            from prompt.math.single_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None,
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
            # print(f"[VALIDATE] Single-turn mode enabled (no FINISH_FLAG)")
            # print(f"[VALIDATE] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            # print(f"[VALIDATE] agent_roles: {rollout_meta_info['agent_roles']}")

        # print(f"\n[VALIDATE] Starting validation loop: {len(self.val_dataloader)} batches")
        # print(f"[VALIDATE] Strategy: Check qa_pairs_json for each sample - if non-empty, conversation has ended and will be evaluated")
        total_batches = len(self.val_dataloader)
        
        for batch_idx, test_data in enumerate(self.val_dataloader):
            # print(f"\n{'*'*80}")
            # print(f"VALIDATION BATCH {batch_idx + 1}/{total_batches}")
            # print(f"{'*'*80}")
            # print(f"\n[VAL BATCH {batch_idx + 1}] Creating batch from dataloader...")
            # print(f"[VAL BATCH {batch_idx + 1}] Batch size: {len(test_data['question'])}")
            # print(f"[VAL BATCH {batch_idx + 1}] test_data keys: {list(test_data.keys())}")
            
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_data['epoch'] = torch.full((len(test_data['question']),), self.global_steps, dtype=torch.long)
            
            # Add validation epoch/split info
            # rollout_meta_info['epoch'] = self.global_steps  # validation epoch
            rollout_meta_info['split'] = 'validation'
            
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)
            # print(f"[VAL BATCH {batch_idx + 1}] test_batch.batch keys: {list(test_batch.batch.keys())}")
            # print(f"[VAL BATCH {batch_idx + 1}] test_batch.non_tensor_batch keys: {list(test_batch.non_tensor_batch.keys())}")

            # Check which samples have finished (non-zero num_questions) BEFORE repeating
            num_questions_list = test_batch.non_tensor_batch['num_qas']
            finished_mask = [num_questions > 0 for num_questions in num_questions_list]
            num_finished = sum(finished_mask)
            # print(f"[VAL BATCH {batch_idx + 1}] Found {num_finished}/{len(finished_mask)} finished conversations (with non-empty qa_pairs_json)")

            # (Generation inputs and ground truths will be collected only for finished conversations)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)
            # print(f"[VAL BATCH {batch_idx + 1}] Repeated batch {self.config.actor_rollout_ref.rollout.val_kwargs.n} times. New size: {len(test_batch.batch)}")
            
            # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
            n_rollouts_val = self.config.actor_rollout_ref.rollout.val_kwargs.n
            rollout_idx = torch.arange(0, n_rollouts_val).repeat(len(test_batch.batch) // n_rollouts_val)
            test_batch.batch['rollout_idx'] = rollout_idx.numpy()
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                # print(f"[VAL BATCH {batch_idx + 1}] Skipping model-based reward model validation")
                return {}

            # print(f"\n[VAL BATCH {batch_idx + 1}] Preparing generation batch...")
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('multi_modal_inputs validation not implemented yet')
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['rollout_idx', 'batch_idx', 'epoch'], 
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json', 'cumulative_session_tokens'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'], 
                        deepcopy=True
                    )
            
            # print(f"[VAL BATCH {batch_idx + 1}] Generation batch prepared with {len(test_gen_batch.batch)} samples")
            # print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.batch keys: {list(test_gen_batch.batch.keys())}")
            # print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.non_tensor_batch keys: {list(test_gen_batch.non_tensor_batch.keys())}")
            # print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.meta_info keys: {list(test_gen_batch.meta_info.keys())}")
            
            test_gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            # print(f'[VAL BATCH {batch_idx + 1}] test_gen_batch meta_info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            # print(f"\n[VAL BATCH {batch_idx + 1}] Padding to be divisible by world_size={self.actor_rollout_wg.world_size}...")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # print(f"[VAL BATCH {batch_idx + 1}] Padded batch size: {len(test_gen_batch_padded.batch)}, pad_size: {pad_size}")
            
            # print(f"[VAL BATCH {batch_idx + 1}] >>> Calling multi_turn_generate_sequences...")
            test_output_gen_batch_padded = self.actor_rollout_wg.multi_turn_generate_sequences(test_gen_batch_padded)
            # print(f"[VAL BATCH {batch_idx + 1}] <<< Generation complete")

            # unpad
            # print(f"[VAL BATCH {batch_idx + 1}] Unpadding batch...")
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print(f'[VAL BATCH {batch_idx + 1}] Validation generation end. Output batch size: {len(test_output_gen_batch.batch)}')

            # (Generated outputs and history will be collected only for finished conversations)
            
            # Collect generation metrics (num_turns, completion_tokens) from ALL batches
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            # print(f"[VAL BATCH {batch_idx + 1}] num_turns (all samples): min={num_turns.min()}, max={num_turns.max()}, mean={num_turns.float().mean()}")
            
            turn_level_completion_tokens = test_output_gen_batch.batch['meta_thinking_num_gen_tokens'].cpu() + \
                test_output_gen_batch.batch['reasoning_num_gen_tokens'].cpu()
            completion_tokens = turn_level_completion_tokens.sum(dim=-1)
            completion_tokens_lst.append(completion_tokens)
            # print(f"[VAL BATCH {batch_idx + 1}] completion_tokens (all samples): min={completion_tokens.min()}, max={completion_tokens.max()}, mean={completion_tokens.float().mean()}")
            
            # Check if any conversations finished in this batch and evaluate them
            if num_finished > 0:
                # print(f"[VAL BATCH {batch_idx + 1}] Evaluating {num_finished} finished conversations...")
                # print(f"[VAL BATCH {batch_idx + 1}] Merging generation output with test batch...")
                test_batch_with_gen = test_batch.union(test_output_gen_batch)
                test_batch_with_gen.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                test_batch_with_gen.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                
                # Filter to only finished conversations (after repeating, mask needs to be replicated)
                repeat_times = self.config.actor_rollout_ref.rollout.val_kwargs.n
                finished_indices = []
                for i, is_finished in enumerate(finished_mask):
                    if is_finished:
                        # Add all repeated versions of this finished sample
                        finished_indices.extend([i * repeat_times + j for j in range(repeat_times)])
                
                # print(f"[VAL BATCH {batch_idx + 1}] Filtering to {len(finished_indices)} samples (finished conversations after repeating)")
                
                # Select only finished samples
                finished_test_batch = test_batch_with_gen[finished_indices]
                
                # Safely collect inputs, outputs, ground truths and history for ONLY finished conversations
                sample_inputs.extend(finished_test_batch.non_tensor_batch['question'])
                sample_outputs.extend(finished_test_batch.non_tensor_batch['response'])
                history_lst.extend(finished_test_batch.non_tensor_batch['history'].tolist())
                gts = [json.loads(x)[0]["answer"] if (isinstance(x, str) and json.loads(x)) else "N/A" for x in finished_test_batch.non_tensor_batch['qa_pairs_json']]
                sample_groundtruths.extend(gts)
                
                # Compute rewards for finished conversations
                # print(f"[VAL BATCH {batch_idx + 1}] Computing rewards for finished conversations...")
                reward_tensor = self.val_reward_fn(finished_test_batch, compression_penalty=self.config.trainer.compression_penalty)
                # print(f"[VAL BATCH {batch_idx + 1}] Reward tensor keys: {list(reward_tensor.keys())}")
                
                reward_tensor_lst.append(reward_tensor['reasoning_turn_level_reward'])
                reward_tensor_dict_lst.append(reward_tensor)  # Store full dict
                acc_tensor_lst.append(reward_tensor['acc'])
                bleu_tensor_lst.append(reward_tensor['bleu'])
                # print(f"[VAL BATCH {batch_idx + 1}] reasoning_turn_level_reward shape: {reward_tensor['reasoning_turn_level_reward'].shape}")
                # print(f"[VAL BATCH {batch_idx + 1}] acc shape: {reward_tensor['acc'].shape}")
                # print(f"[VAL BATCH {batch_idx + 1}] bleu shape: {reward_tensor['bleu'].shape}")
                
                # Store scores
                scores = reward_tensor['reasoning_turn_level_reward'].sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                # print(f"[VAL BATCH {batch_idx + 1}] Sample scores[0]: {scores[0]}")
                
                # Get data sources from finished samples
                data_source_lst.append(finished_test_batch.non_tensor_batch.get('subset', ['locomo'] * reward_tensor['reasoning_turn_level_reward'].shape[0]))
            # else:
                # print(f"[VAL BATCH {batch_idx + 1}] No finished conversations in this batch, skipping reward computation")
            
            # print(f"[VAL BATCH {batch_idx + 1}] Batch processing complete\n")

        # Now compute final metrics from all finished conversations across all batches
        # print(f"\n[VALIDATE] All batches processed. Computing final metrics...")
        
        if len(reward_tensor_lst) == 0:
            print("[VALIDATE] WARNING: No finished conversations found across all batches!")
            return {}
        
        # Log generations
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, groundtruths=sample_groundtruths, histories=history_lst)

        # Concatenate all accumulated tensors
        # print(f"[VALIDATE] Concatenating reward tensors from {len(reward_tensor_lst)} batches with finished conversations...")
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (total_finished_samples,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        bleu_tensor = torch.cat(bleu_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # print(f"[VALIDATE] Total finished samples evaluated: {reward_tensor.shape[0]}")
        # print(f"[VALIDATE] Mean reward: {reward_tensor.mean().item():.4f}")
        # print(f"[VALIDATE] Mean accuracy: {acc_tensor.mean().item():.4f}")
        # print(f"[VALIDATE] Mean BLEU: {bleu_tensor.mean().item():.4f}")

        # Compute metrics for all finished conversations
        data_source_reward = {}
        data_source_acc = {}
        data_source_bleu = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            if data_source not in data_source_acc:
                data_source_acc[data_source] = []
            data_source_acc[data_source].append(acc_tensor[i].item())
            if data_source not in data_source_bleu:
                data_source_bleu[data_source] = []
            data_source_bleu[data_source].append(bleu_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            # print(f"[VALIDATE] {data_source} mean reward: {np.mean(rewards):.4f}")
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)
            # print(f"[VALIDATE] {data_source} mean accuracy: {np.mean(accs):.4f}")
        for data_source, bleus in data_source_bleu.items():
            metric_dict[f'val/bleu/{data_source}'] = np.mean(bleus)
            # print(f"[VALIDATE] {data_source} mean BLEU: {np.mean(bleus):.4f}")
        
        # Stage 2 aggregation: Combine per-category metrics across ALL validation batches
        # Each batch returns sum and count (not averages), so we just accumulate them
        # print(f"\n[VALIDATE] Aggregating per-category metrics across {len(reward_tensor_dict_lst)} batches...")
        if len(reward_tensor_dict_lst) > 0:
            category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial']
            category_aggregates = {}
            
            # Accumulate sums and counts from all batches (no multiplication needed!)
            for batch_idx, reward_dict in enumerate(reward_tensor_dict_lst):
                for cat_name in category_names:
                    f1_sum_key = f'{cat_name}_f1_sum'
                    if f1_sum_key in reward_dict:
                        if cat_name not in category_aggregates:
                            category_aggregates[cat_name] = {'f1_sum': 0.0, 'bleu_sum': 0.0, 'count': 0}
                        
                        # Directly accumulate sums and counts
                        category_aggregates[cat_name]['f1_sum'] += reward_dict[f1_sum_key].item()
                        category_aggregates[cat_name]['bleu_sum'] += reward_dict[f'{cat_name}_bleu_sum'].item()
                        category_aggregates[cat_name]['count'] += int(reward_dict[f'{cat_name}_count'].item())
            
            # Compute global averages (only once, at the end)
            if len(category_aggregates) > 0:
                # print(f"[VALIDATE] Found {len(category_aggregates)} categories with data")
                for cat_name in sorted(category_aggregates.keys()):
                    agg = category_aggregates[cat_name]
                    if agg['count'] > 0:
                        metric_dict[f'val/{cat_name}_f1'] = agg['f1_sum'] / agg['count']
                        metric_dict[f'val/{cat_name}_bleu'] = agg['bleu_sum'] / agg['count']
                        metric_dict[f'val/{cat_name}_count'] = agg['count']
                        # print(f"[VALIDATE] {cat_name}: F1={metric_dict[f'val/{cat_name}_f1']:.4f}, BLEU={metric_dict[f'val/{cat_name}_bleu']:.4f}, count={metric_dict[f'val/{cat_name}_count']:.0f}")
            # else:
                # print(f"[VALIDATE] Warning: No category data found in any batch")
        # else:
            # print(f"[VALIDATE] No batches with category data")
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['val/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['val/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['val/num_turns/min'] = num_turns_tensor.min().item()
            # print(f"[VALIDATE] num_turns: mean={metric_dict['val/num_turns/mean']:.2f}, max={metric_dict['val/num_turns/max']}, min={metric_dict['val/num_turns/min']}")
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['val/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['val/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['val/completion_tokens/min'] = completion_tokens_tensor.min().item()
            # print(f"[VALIDATE] completion_tokens: mean={metric_dict['val/completion_tokens/mean']:.2f}, max={metric_dict['val/completion_tokens/max']}, min={metric_dict['val/completion_tokens/min']}")

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            # print(f"\n[VALIDATE] Saving validation generations...")
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'val_step_{self.global_steps}.jsonl'
            
            results_to_save = []
            for inp, outp, gt, hist, score in zip(sample_inputs, sample_outputs, sample_groundtruths, history_lst, sample_scores):
                unpad_history = [x for x in hist if x['role'] != 'padding']
                results_to_save.append({
                    'question': inp,
                    'answer': outp, 
                    'groundtruth': gt,
                    'history': unpad_history,
                    'score': score
                })
            
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(results_to_save)
            # print(f"[VALIDATE] Saved {len(results_to_save)} validation results to {output_file}")

        # print(f"\n[VALIDATE] Validation complete. Final metrics: {metric_dict}")
        # print("="*80 + "\n")
        return metric_dict

    def _test(self):
        """Test pipeline - runs multi-turn generation on all batches and evaluates QA only for finished conversations (non-empty qa_pairs_json)"""
        # print("\n" + "="*80)
        # print("STARTING TEST")
        # print("="*80)
        
        sample_groundtruths = []
        sample_inputs = []
        sample_outputs = []
        history_lst = []
        
        # Accumulate rewards/metrics for all finished conversations across batches
        reward_tensor_lst = []
        reward_tensor_dict_lst = []  # Store full dicts to access category metrics
        acc_tensor_lst = []
        bleu_tensor_lst = []
        data_source_lst = []
        num_turns_lst = []
        completion_tokens_lst = []
        sample_scores = []

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        single_agent_mode = self.config.actor_rollout_ref.rollout.get('single_agent_mode', False)
        # print(f"\n[TEST] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT, SINGLE_AGENT_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None,
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': SINGLE_AGENT_PROMPT if single_agent_mode else MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
        else:
            from prompt.math.single_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None,
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
            # print(f"[TEST] Single-turn mode enabled")
            # print(f"[TEST] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            # print(f"[TEST] agent_roles: {rollout_meta_info['agent_roles']}")

        # print(f"\n[TEST] Starting test loop: {len(self.test_dataloader)} batches")

        # print(f"[TEST] Strategy: Check qa_pairs_json for each sample - if non-empty, conversation has ended and will be evaluated")
        total_batches = len(self.test_dataloader)
        
        for batch_idx, test_data in enumerate(self.test_dataloader):
            # print(f"\n{'*'*80}")
            # print(f"TEST BATCH {batch_idx + 1}/{total_batches}")
            # print(f"{'*'*80}")
            # print(f"\n[TEST BATCH {batch_idx + 1}] Creating batch from dataloader...")
            # print(f"[TEST BATCH {batch_idx + 1}] Batch size: {len(test_data['question'])}")
            # print(f"[TEST BATCH {batch_idx + 1}] test_data keys: {list(test_data.keys())}")
            
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_data['epoch'] = torch.full((len(test_data['question']),), self.global_steps, dtype=torch.long)
            
            # Add test epoch/split info
            # rollout_meta_info['epoch'] = self.global_steps
            rollout_meta_info['split'] = 'test'
            
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)
            # print(f"[TEST BATCH {batch_idx + 1}] test_batch.batch keys: {list(test_batch.batch.keys())}")
            # print(f"[TEST BATCH {batch_idx + 1}] test_batch.non_tensor_batch keys: {list(test_batch.non_tensor_batch.keys())}")

            # Check which samples have finished (non-zero num_questions) BEFORE repeating
            num_questions_list = test_batch.non_tensor_batch['num_qas']
            finished_mask = [num_questions > 0 for num_questions in num_questions_list]
            num_finished = sum(finished_mask)
            # print(f"[TEST BATCH {batch_idx + 1}] Found {num_finished}/{len(finished_mask)} finished conversations (with non-empty qa_pairs_json)")

            # (Generation inputs and ground truths will be collected only for finished conversations)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)
            # print(f"[TEST BATCH {batch_idx + 1}] Repeated batch {self.config.actor_rollout_ref.rollout.val_kwargs.n} times. New size: {len(test_batch.batch)}")
            
            # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
            n_rollouts_test = self.config.actor_rollout_ref.rollout.val_kwargs.n
            rollout_idx = torch.arange(0, n_rollouts_test).repeat(len(test_batch.batch) // n_rollouts_test)
            test_batch.batch['rollout_idx'] = rollout_idx.numpy()
            
            # we only do test on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                # print(f"[TEST BATCH {batch_idx + 1}] Skipping model-based reward model test")
                return {}

            # print(f"\n[TEST BATCH {batch_idx + 1}] Preparing generation batch...")
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('multi_modal_inputs test not implemented yet')
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['rollout_idx', 'batch_idx', 'epoch'], 
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json', 'cumulative_session_tokens'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'], 
                        deepcopy=True
                    )
            
            # print(f"[TEST BATCH {batch_idx + 1}] Generation batch prepared with {len(test_gen_batch.batch)} samples")
            
            test_gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            # print(f'[TEST BATCH {batch_idx + 1}] test_gen_batch meta_info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            # print(f"\n[TEST BATCH {batch_idx + 1}] Padding to be divisible by world_size={self.actor_rollout_wg.world_size}...")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # print(f"[TEST BATCH {batch_idx + 1}] Padded batch size: {len(test_gen_batch_padded.batch)}, pad_size: {pad_size}")
            
            # print(f"[TEST BATCH {batch_idx + 1}] >>> Calling multi_turn_generate_sequences...")
            test_output_gen_batch_padded = self.actor_rollout_wg.multi_turn_generate_sequences(test_gen_batch_padded)
            # print(f"[TEST BATCH {batch_idx + 1}] <<< Generation complete")

            # unpad
            # print(f"[TEST BATCH {batch_idx + 1}] Unpadding batch...")
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print(f'[TEST BATCH {batch_idx + 1}] Test generation end. Output batch size: {len(test_output_gen_batch.batch)}')

            # (Generated outputs and history will be collected only for finished conversations)
            
            # Collect generation metrics (num_turns, completion_tokens) from ALL batches
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            print(f"[TEST BATCH {batch_idx + 1}] num_turns (all samples): min={num_turns.min()}, max={num_turns.max()}, mean={num_turns.float().mean()}")
            
            turn_level_completion_tokens = test_output_gen_batch.batch['meta_thinking_num_gen_tokens'].cpu() + \
                test_output_gen_batch.batch['reasoning_num_gen_tokens'].cpu()
            completion_tokens = turn_level_completion_tokens.sum(dim=-1)
            completion_tokens_lst.append(completion_tokens)
            # print(f"[TEST BATCH {batch_idx + 1}] completion_tokens (all samples): min={completion_tokens.min()}, max={completion_tokens.max()}, mean={completion_tokens.float().mean()}")
            
            # Check if any conversations finished in this batch and evaluate them
            if num_finished > 0:
                # print(f"[TEST BATCH {batch_idx + 1}] Evaluating {num_finished} finished conversations...")
                # print(f"[TEST BATCH {batch_idx + 1}] Merging generation output with test batch...")
                test_batch_with_gen = test_batch.union(test_output_gen_batch)
                test_batch_with_gen.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                test_batch_with_gen.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                
                # Filter to only finished conversations (after repeating, mask needs to be replicated)
                repeat_times = self.config.actor_rollout_ref.rollout.val_kwargs.n
                finished_indices = []
                for i, is_finished in enumerate(finished_mask):
                    if is_finished:
                        # Add all repeated versions of this finished sample
                        finished_indices.extend([i * repeat_times + j for j in range(repeat_times)])
                
                # print(f"[TEST BATCH {batch_idx + 1}] Filtering to {len(finished_indices)} samples (finished conversations after repeating)")
                
                # Select only finished samples
                finished_test_batch = test_batch_with_gen[finished_indices]
                
                # Safely collect inputs, outputs, ground truths and history for ONLY finished conversations
                sample_inputs.extend(finished_test_batch.non_tensor_batch['question'])
                sample_outputs.extend(finished_test_batch.non_tensor_batch['response'])
                history_lst.extend(finished_test_batch.non_tensor_batch['history'].tolist())
                gts = [json.loads(x)[0]["answer"] if (isinstance(x, str) and json.loads(x)) else "N/A" for x in finished_test_batch.non_tensor_batch['qa_pairs_json']]
                sample_groundtruths.extend(gts)
                
                # Compute rewards for finished conversations
                # print(f"[TEST BATCH {batch_idx + 1}] Computing rewards for finished conversations...")
                reward_tensor = self.val_reward_fn(finished_test_batch, compression_penalty=self.config.trainer.compression_penalty)
                # print(f"[TEST BATCH {batch_idx + 1}] Reward tensor keys: {list(reward_tensor.keys())}")
                
                reward_tensor_lst.append(reward_tensor['reasoning_turn_level_reward'])
                reward_tensor_dict_lst.append(reward_tensor)  # Store full dict
                acc_tensor_lst.append(reward_tensor['acc'])
                bleu_tensor_lst.append(reward_tensor['bleu'])
                # print(f"[TEST BATCH {batch_idx + 1}] reasoning_turn_level_reward shape: {reward_tensor['reasoning_turn_level_reward'].shape}")
                # print(f"[TEST BATCH {batch_idx + 1}] acc shape: {reward_tensor['acc'].shape}")
                # print(f"[TEST BATCH {batch_idx + 1}] bleu shape: {reward_tensor['bleu'].shape}")
                
                # Store scores
                scores = reward_tensor['reasoning_turn_level_reward'].sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                # print(f"[TEST BATCH {batch_idx + 1}] Sample scores[0]: {scores[0]}")
                
                # Get data sources from finished samples
                data_source_lst.append(finished_test_batch.non_tensor_batch.get('subset', ['locomo'] * reward_tensor['reasoning_turn_level_reward'].shape[0]))
            # else:
                # print(f"[TEST BATCH {batch_idx + 1}] No finished conversations in this batch, skipping reward computation")
            
            # print(f"[TEST BATCH {batch_idx + 1}] Batch processing complete\n")

        # Now compute final metrics from all finished conversations across all batches
        # print(f"\n[TEST] All batches processed. Computing final metrics...")
        
        if len(reward_tensor_lst) == 0:
            # print("[TEST] WARNING: No finished conversations found across all batches!")
            return {}
        
        # Log generations
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, groundtruths=sample_groundtruths, histories=history_lst)

        # Concatenate all accumulated tensors
        # print(f"[TEST] Concatenating reward tensors from {len(reward_tensor_lst)} batches with finished conversations...")
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (total_finished_samples,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        bleu_tensor = torch.cat(bleu_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # print(f"[TEST] Total finished samples evaluated: {reward_tensor.shape[0]}")
        # print(f"[TEST] Mean reward: {reward_tensor.mean().item():.4f}")
        # print(f"[TEST] Mean accuracy: {acc_tensor.mean().item():.4f}")
        # print(f"[TEST] Mean BLEU: {bleu_tensor.mean().item():.4f}")

        # Compute metrics for all finished conversations
        data_source_reward = {}
        data_source_acc = {}
        data_source_bleu = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            if data_source not in data_source_acc:
                data_source_acc[data_source] = []
            data_source_acc[data_source].append(acc_tensor[i].item())
            if data_source not in data_source_bleu:
                data_source_bleu[data_source] = []
            data_source_bleu[data_source].append(bleu_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test/test_score/{data_source}'] = np.mean(rewards)
            # print(f"[TEST] {data_source} mean reward: {np.mean(rewards):.4f}")
        for data_source, accs in data_source_acc.items():
            metric_dict[f'test/acc/{data_source}'] = np.mean(accs)
            # print(f"[TEST] {data_source} mean accuracy: {np.mean(accs):.4f}")
        for data_source, bleus in data_source_bleu.items():
            metric_dict[f'test/bleu/{data_source}'] = np.mean(bleus)
            # print(f"[TEST] {data_source} mean BLEU: {np.mean(bleus):.4f}")
        
        # Stage 2 aggregation: Combine per-category metrics across ALL test batches
        # Each batch returns sum and count (not averages), so we just accumulate them
        # print(f"\n[TEST] Aggregating per-category metrics across {len(reward_tensor_dict_lst)} batches...")
        if len(reward_tensor_dict_lst) > 0:
            category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial']
            category_aggregates = {}
            
            # Accumulate sums and counts from all batches (no multiplication needed!)
            for batch_idx, reward_dict in enumerate(reward_tensor_dict_lst):
                for cat_name in category_names:
                    f1_sum_key = f'{cat_name}_f1_sum'
                    if f1_sum_key in reward_dict:
                        if cat_name not in category_aggregates:
                            category_aggregates[cat_name] = {'f1_sum': 0.0, 'bleu_sum': 0.0, 'count': 0}
                        
                        # Directly accumulate sums and counts
                        category_aggregates[cat_name]['f1_sum'] += reward_dict[f1_sum_key].item()
                        category_aggregates[cat_name]['bleu_sum'] += reward_dict[f'{cat_name}_bleu_sum'].item()
                        category_aggregates[cat_name]['count'] += int(reward_dict[f'{cat_name}_count'].item())
            
            # Compute global averages (only once, at the end)
            if len(category_aggregates) > 0:
                # print(f"[TEST] Found {len(category_aggregates)} categories with data")
                for cat_name in sorted(category_aggregates.keys()):
                    agg = category_aggregates[cat_name]
                    if agg['count'] > 0:
                        metric_dict[f'test/{cat_name}_f1'] = agg['f1_sum'] / agg['count']
                        metric_dict[f'test/{cat_name}_bleu'] = agg['bleu_sum'] / agg['count']
                        metric_dict[f'test/{cat_name}_count'] = agg['count']
                        # print(f"[TEST] {cat_name}: F1={metric_dict[f'test/{cat_name}_f1']:.4f}, BLEU={metric_dict[f'test/{cat_name}_bleu']:.4f}, count={metric_dict[f'test/{cat_name}_count']:.0f}")
            # else:
                # print(f"[TEST] Warning: No category data found in any batch")
        # else:
            # print(f"[TEST] No batches with category data")
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['test/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['test/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['test/num_turns/min'] = num_turns_tensor.min().item()
            # print(f"[TEST] num_turns: mean={metric_dict['test/num_turns/mean']:.2f}, max={metric_dict['test/num_turns/max']}, min={metric_dict['test/num_turns/min']}")
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['test/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['test/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['test/completion_tokens/min'] = completion_tokens_tensor.min().item()
            # print(f"[TEST] completion_tokens: mean={metric_dict['test/completion_tokens/mean']:.2f}, max={metric_dict['test/completion_tokens/max']}, min={metric_dict['test/completion_tokens/min']}")

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            # print(f"\n[TEST] Saving test generations...")
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'test_step_{self.global_steps}.jsonl'
            
            results_to_save = []
            for inp, outp, gt, hist, score in zip(sample_inputs, sample_outputs, sample_groundtruths, history_lst, sample_scores):
                unpad_history = [x for x in hist if x['role'] != 'padding']
                results_to_save.append({
                    'question': inp,
                    'answer': outp, 
                    'groundtruth': gt,
                    'history': unpad_history,
                    'score': score
                })
            
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(results_to_save)
            # print(f"[TEST] Saved {len(results_to_save)} test results to {output_file}")

        # print(f"\n[TEST] Test complete. Final metrics: {metric_dict}")
        # print("="*80 + "\n")
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # attention_mask = batch.batch['attention_mask']
        meta_thinking_attention_mask = batch.batch['meta_thinking_attention_mask']
        reasoning_attention_mask = batch.batch['reasoning_attention_mask']
        batch_size = meta_thinking_attention_mask.shape[0]
        global_seqlen_lst = (meta_thinking_attention_mask.view(batch_size, -1).sum(-1) + reasoning_attention_mask.view(batch_size, -1).sum(-1)).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        # print("\n" + "="*80)
        # print("STARTING PPO TRAINING LOOP (fit method)")
        # print("="*80)

        self.global_steps = 0
        
        # Best validation tracking
        self.best_val_acc = -1.0
        self.best_global_step = -1
        self.patience_counter = 0

        # load checkpoint before doing anything
        # print("\n[FIT] Loading checkpoint...")
        self._load_checkpoint()
        # print(f"[FIT] Checkpoint loaded. Starting from global_steps={self.global_steps}")

        if self.config.trainer.get('fork_wandb_id', None) is not None:
            fork_wandb_id = self.config.trainer.fork_wandb_id
            # wandb_kwargs = {'resume': 'must', 'id': fork_wandb_id}
            # print(f'**[WANDB]: will fork run from wandb id: `{fork_wandb_id}` at step {self.global_steps} **')
            
            # e.g. fork_from="6yaq69uw?_step=200"
            wandb_kwargs = {'fork_from': f"{fork_wandb_id}?_step={self.global_steps}"}
        else:
            wandb_kwargs = {}
        
        # print(f"\n[FIT] Initializing logger: {self.config.trainer.logger}")
        # print(f"[FIT] Project: {self.config.trainer.project_name}, Experiment: {self.config.trainer.experiment_name}")
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_kwargs=wandb_kwargs
                          )

        # perform test if test_only mode is enabled
        if self.val_reward_fn is not None and self.config.trainer.get('test_only', False):
            # print("\n[FIT] Test-only mode enabled. Running test and exiting...")
            test_metrics = self._test()
            pprint(f'Test metrics: {test_metrics}')
            logger.log(data=test_metrics, step=self.global_steps)
            # print("[FIT] Test-only mode complete. Exiting.")
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # print("\n[FIT] Running initial validation before training...")
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                # print("[FIT] Val-only mode enabled. Exiting after validation.")
                return

        # perform test evaluation before training
        if self.test_dataloader is not None and self.config.trainer.get('test_before_train', False):
            test_metrics = self._test()
            pprint(f'Initial test metrics: {test_metrics}')
            logger.log(data=test_metrics, step=self.global_steps)

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        single_agent_mode = self.config.actor_rollout_ref.rollout.get('single_agent_mode', False)
        # print(f"\n[FIT] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT, SINGLE_AGENT_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None,
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': SINGLE_AGENT_PROMPT if single_agent_mode else MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
        else:
            from prompt.math.single_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None,
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
            # print(f"[FIT] Single-turn mode enabled (no FINISH_FLAG)")
            # print(f"[FIT] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            # print(f"[FIT] agent_roles: {rollout_meta_info['agent_roles']}")
        
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_prompt_cnt = 0 
        all_negative_cnt = 0
        all_positive_cnt = 0
        kept_prompt_cnt = 0

        # print(f"\n[FIT] Starting training loop: {self.config.trainer.total_epochs} epochs, {len(self.train_dataloader)} batches per epoch")
        # print(f"[FIT] Total training steps: {self.total_training_steps}")

        for epoch in range(self.config.trainer.total_epochs):
            # print(f"\n{'='*80}")
            # print(f"EPOCH {epoch + 1}/{self.config.trainer.total_epochs}")
            # print(f"{'='*80}")
            
            # --- Initialize Inner Sampling Buffers ---
            inner_sampling_fraction = self.config.actor_rollout_ref.rollout.get('inner_sampling_fraction', 0.0)
            inner_n = self.config.actor_rollout_ref.rollout.get('inner_n', 1)
            inner_sampling_buffer = []
            evaluated_inner_batches = []
            
            for batch_dict in self.train_dataloader:
                # print(f"\n{'*'*80}")
                # print(f"TRAINING STEP {self.global_steps}/{self.total_training_steps}")
                # print(f"{'*'*80}")
                metrics = {}
                timing_raw = {}

                # print("batch_dict[chunk_id]: ", batch_dict['chunk_id'] if 'chunk_id' in batch_dict else 'N/A')
                # print("batch_dict[sample_id]: ", batch_dict['sample_id'] if 'sample_id' in batch_dict else 'N/A')

                # create a dummy tensor for the construction function
                # print(f"\n[STEP {self.global_steps}] Creating batch from dataloader...")
                # print(f"[STEP {self.global_steps}] Batch size: {len(batch_dict['question'])}")
                # print(f"[STEP {self.global_steps}] batch_dict keys: {list(batch_dict.keys())}")
                
                dummy_tensor = torch.arange(0, len(batch_dict['question']))
                batch_dict['batch_idx'] = dummy_tensor
                batch_dict['epoch'] = torch.full((len(batch_dict['question']),), epoch, dtype=torch.long)

                # --- simple index-based replay buffer (lightweight) ---
                # instantiate once
                if not hasattr(self, 'replay_buffer'):
                    capacity = self.config.trainer.get('replay_buffer_size', 100)
                    self.replay_buffer = ReplayBuffer(capacity=capacity)

                # add current dataset indices to buffer (dataset provides 'index' per sample)
                if 'index' in batch_dict:
                    try:
                        idxs = batch_dict['index'].tolist() if isinstance(batch_dict['index'], np.ndarray) else list(batch_dict['index'])
                        # print(f"[STEP {self.global_steps}] Adding {len(idxs)} indices to replay buffer (step={self.global_steps})")
                        # print(f"[STEP {self.global_steps}] idxs: {idxs}")
                        self.replay_buffer.add_indices(idxs, epoch=epoch)
                    except Exception:
                        pass

                # sample from replay and merge with current batch if requested
                replay_ratio = float(self.config.trainer.get('replay_mix_ratio', 0.0))
                n_replay = int(len(batch_dict['question']) * replay_ratio)
                if n_replay > 0 and getattr(self, 'replay_buffer', None) and len(self.replay_buffer.buffer) > 0:
                    sampled = self.replay_buffer.sample(n_replay, strategy=self.config.trainer.get('replay_strategy', 'uniform'))
                    # print(f"[STEP {self.global_steps}] Sampling {len(sampled)} entries from replay buffer to merge into current batch")
                    if len(sampled) > 0:
                        # sampled contains tuples (index, orig_epoch)
                        sample_indices = [s[0] for s in sampled]
                        sample_epochs = [s[1] for s in sampled]
                        # print(f"[STEP {self.global_steps}] sample_indices: {sample_indices}, sample_epochs: {sample_epochs}")
                        # get raw items and collate using project's collate_fn
                        replay_items = [self.train_dataset[int(i)] for i in sample_indices]
                        try:
                            replay_batch = collate_fn(replay_items)
                            # Ensure per-sample provenance tensors exist for replayed items so
                            # merged dicts have consistent batch dims. Use stored orig_epoch.
                            if len(replay_batch) > 0:
                                first_key = list(replay_batch.keys())[0]
                                first_val = replay_batch[first_key]
                                if isinstance(first_val, torch.Tensor):
                                    n_replay_actual = first_val.shape[0]
                                elif isinstance(first_val, np.ndarray):
                                    n_replay_actual = first_val.shape[0]
                                else:
                                    n_replay_actual = len(first_val)

                                # attach stored epoch provenance
                                replay_batch['epoch'] = torch.tensor(sample_epochs, dtype=torch.long)
                                # mark replayed samples (non-tensor field)
                                # replay_batch['is_replay'] = np.array([True] * n_replay_actual, dtype=object)
                                # optionally add replay_age
                                # try:
                                #     replay_batch['replay_age'] = (epoch - torch.tensor(sample_epochs, dtype=torch.long)).tolist()
                                # except Exception:
                                #     pass

                            merged = merge_batch_dicts([batch_dict, replay_batch])
                            # recompute batch_idx for merged batch
                            merged_size = merged[list(merged.keys())[0]].shape[0]
                            merged['batch_idx'] = torch.arange(0, merged_size)
                            batch_dict = merged
                        except Exception:
                            # fallback: ignore replay if anything goes wrong
                            pass

                # Add some meta_info to be used in rollouts
                rollout_meta_info['split'] = 'train'

                new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=rollout_meta_info)
                # print(f"[STEP {self.global_steps}] new_batch.batch keys: {list(new_batch.batch.keys())}")
                # print(f"[STEP {self.global_steps}] new_batch.non_tensor_batch keys: {list(new_batch.non_tensor_batch.keys())}")
                # print(f"[STEP {self.global_steps}] new_batch.meta_info keys: {list(new_batch.meta_info.keys())}")
                # if 'batch_idx' in new_batch.batch:
                    # print(f"[STEP {self.global_steps}] batch_idx shape: {new_batch.batch['batch_idx'].shape}")
                new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
                # Ensure rollout_idx is consistent per conversation (0..n-1) specifically for memory caching
                # This ensures that even if batch order changes, the i-th rollout of a conversation always uses the same index.
                n_rollouts = self.config.actor_rollout_ref.rollout.n
                # If interleave=True: [A0, A1... An-1, B0, B1... Bn-1]
                # We want indices: [0, 1... n-1, 0, 1... n-1]
                rollout_idx = torch.arange(0, n_rollouts).repeat(len(new_batch.batch) // n_rollouts)
                new_batch.batch['rollout_idx'] = rollout_idx.numpy()

                num_gen_batches += 1

                # pop those keys for generation
                # print(f"\n[STEP {self.global_steps}] Preparing generation batch...")
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    raise NotImplementedError('multi_modal_inputs is not implemented yet')
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    # because verl originally calls this 'chat'
                    gen_batch = new_batch.select(
                        batch_keys=['rollout_idx', 'batch_idx', 'epoch'], 
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json', 'cumulative_session_tokens'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'],
                        deepcopy=True
                    )
                
                # --- INNER SAMPLING COLLECTION PHASE ---
                if inner_sampling_fraction > 0.0:
                    import random
                    # Decide at the SESSION level whether to do inner sampling
                    if random.random() < inner_sampling_fraction:
                        # We decided to inner-sample this session!
                        # Now, for each unique question/sample in the batch, pick EXACTLY ONE 
                        # of its parallel rollouts to branch off of for inner GRPO.
                        sampled_indices = []
                        batch_idxs = gen_batch.batch['batch_idx']
                        unique_bidxs = np.unique(batch_idxs)
                        
                        for bidx in unique_bidxs:
                            # Find all parallel rollouts for this specific batch item
                            idx_for_bidx = np.where(batch_idxs == bidx)[0]
                            if len(idx_for_bidx) > 0:
                                sampled_indices.append(random.choice(idx_for_bidx))
                                
                        if len(sampled_indices) > 0:
                            inner_item = new_batch[sampled_indices]
                            # Assign completely new uids for these inner samples to isolate them
                            inner_item.non_tensor_batch['uid'] = np.array([f"inner_{uuid.uuid4()}" for _ in range(len(inner_item.batch))], dtype=object)
                            # Keep the selected normal rollout index for memory loading.
                            if 'rollout_idx' in inner_item.batch:
                                source_rollout_idx = inner_item.batch['rollout_idx']
                                if isinstance(source_rollout_idx, torch.Tensor):
                                    inner_item.batch['source_rollout_idx'] = source_rollout_idx.clone()
                                else:
                                    inner_item.batch['source_rollout_idx'] = np.copy(source_rollout_idx)
                            # Remove or reset rollout_idx because we will repeat them for inner_n
                            if 'rollout_idx' in inner_item.batch:
                                inner_item.batch.pop('rollout_idx')
                            inner_sampling_buffer.append(inner_item)
                # print(f"[STEP {self.global_steps}] Generation batch prepared with {len(gen_batch.batch)} samples")
                # print(f"[STEP {self.global_steps}] gen_batch.batch keys: {list(gen_batch.batch.keys())}")
                # print(f"[STEP {self.global_steps}] gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
                # print(f"[STEP {self.global_steps}] gen_batch.meta_info keys: {list(gen_batch.meta_info.keys())}")

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    # print(f"\n[STEP {self.global_steps}] >>> Calling multi_turn_generate_sequences...")
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.multi_turn_generate_sequences(gen_batch)
                    # print(f"[STEP {self.global_steps}] <<< Generation complete. Output batch size: {len(gen_batch_output.batch)}")
                    # print(f"[STEP {self.global_steps}] gen_batch_output.batch keys: {list(gen_batch_output.batch.keys())}")
                    # print(f"[STEP {self.global_steps}] gen_batch_output.non_tensor_batch keys: {list(gen_batch_output.non_tensor_batch.keys())}")
                    # if 'meta_thinking_attention_mask' in gen_batch_output.batch:
                        # print(f"[STEP {self.global_steps}] meta_thinking_attention_mask shape: {gen_batch_output.batch['meta_thinking_attention_mask'].shape}")
                    # if 'reasoning_attention_mask' in gen_batch_output.batch:
                        # print(f"[STEP {self.global_steps}] reasoning_attention_mask shape: {gen_batch_output.batch['reasoning_attention_mask'].shape}")
                    # if 'response' in gen_batch_output.non_tensor_batch:
                        # print(f"[STEP {self.global_steps}] Sample response[0]: {gen_batch_output.non_tensor_batch['response'][0][:100] if isinstance(gen_batch_output.non_tensor_batch['response'][0], str) else gen_batch_output.non_tensor_batch['response'][0]}...")
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError('REMAX is not implemented yet')
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch, compression_penalty=self.config.trainer.compression_penalty)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # print(f"\n[STEP {self.global_steps}] Merging generation output with original batch...")
                    # print(f"[STEP {self.global_steps}] BEFORE union - gen_batch_output.batch keys: {list(gen_batch_output.batch.keys())}")
                    new_batch = new_batch.union(gen_batch_output)
                    # print(f"[STEP {self.global_steps}] AFTER union - Merged batch size: {len(new_batch.batch)}")
                    # print(f"[STEP {self.global_steps}] AFTER union - Merged new_batch.batch keys: {list(new_batch.batch.keys())}")
                    # print(f"[STEP {self.global_steps}] Merged new_batch.non_tensor_batch keys: {list(new_batch.non_tensor_batch.keys())}")
                    # if 'num_turns' in new_batch.non_tensor_batch:
                        # print(f"[STEP {self.global_steps}] Sample num_turns[0]: {new_batch.non_tensor_batch['num_turns'][0]}")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        # print(f"[STEP {self.global_steps}] Balancing batch across DP ranks...")
                        self._balance_batch(new_batch, metrics=metrics)

                    # compute global_valid tokens
                    new_batch.meta_info['global_token_num'] = torch.sum(
                        new_batch.batch['meta_thinking_attention_mask'] 
                        + new_batch.batch['reasoning_attention_mask'], 
                        dim=-1
                    ).tolist()

                    # # recompute old_log_probs
                    # with _timer('old_log_prob', timing_raw):
                    #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    #     batch = batch.union(old_log_prob)

                    batch = new_batch

                    # NOTE: critic values are computed after merge_roles_data below

                    # Accumulation Logic
                    current_chunk_id = batch.non_tensor_batch['chunk_id'][0] if 'chunk_id' in batch.non_tensor_batch else 1
                    
                    # Convert to int just in case
                    if isinstance(current_chunk_id, torch.Tensor):
                        current_chunk_id = current_chunk_id.item()
                    
                    # print(f"[STEP {self.global_steps}] chunk_id={current_chunk_id}")

                    # so if max session is set to infinity to match all convs without setting manually,
                    # It will work and accumulate till the last conv !
                    total_chunks = len(self.train_dataloader)
                    if current_chunk_id < total_chunks:
                        # print(f"[STEP {self.global_steps}] Accumulating batch (session {current_chunk_id}, total_chunks={total_chunks})...")
                        self.accumulated_batches.append(batch)
                        # print(f"[STEP {self.global_steps}] Accumulated batches count: {len(self.accumulated_batches)}")
                        # Skip update, continue to next batch (which should be next session)
                        # But we must ensure global_steps is handled correctly. 
                        # If we continue here, global_steps increments at end of loop. 
                        # Maybe we should NOT increment global_steps for accumulated steps? 
                        # Or increment it but don't log/save?
                        # The user wants "epoch level update". Usually 1 update per 5 sessions.
                        # So we can let global_steps increment, but only do update on the 5th step.
                        continue
                    
                    # If we are here, it matches the final session 
                    # print(f"[STEP {self.global_steps}] Final session ({current_chunk_id}) reached. Proceeding to update.")
                    # print(f"[STEP {self.global_steps}] Final available session ({current_chunk_id}) reached. Proceeding to update.")

                    # --- 1. Construct Terminal Batch ---
                    # Combine all batches to find the terminal state of every conversation
                    all_batches = self.accumulated_batches + [batch]
                    # print(f"[STEP {self.global_steps}] Constructing terminal_batch from {len(all_batches)} batches...")
                    
                    # Track the latest occurrence of each (sample_id, rollout_idx)
                    # key: (sample_id, rollout_idx) -> value: (batch_index_in_all_batches, row_index_in_batch)
                    latest_seen = {}
                    
                    # 1.1 Find the Latest Occurrence (Terminal State)
                    # Iterate through all batches (accumulated + current) to identify the last time we saw each conversation.
                    # This handles variable length sessions: the "terminal state" is simply the last available snapshot.
                    for b_idx, b in enumerate(all_batches):
                        chunk_id = b.non_tensor_batch['chunk_id'][0] # All items in a batch have the same chunk_id
                        sample_ids = b.non_tensor_batch['sample_id']
                        rollout_idxs = b.batch['rollout_idx']
                        
                        # Ensure rollout_idxs is a list for consistent indexing
                        if isinstance(rollout_idxs, torch.Tensor):
                            rollout_idxs = rollout_idxs.cpu().tolist()
                        
                        # Map each unique conversation key (sample_id, rollout_idx) to its location (batch_idx, row_idx).
                        # Since we iterate batches in chronological order, later occurrences overwrite earlier ones.
                        # The final value in `latest_seen` is guaranteed to be the terminal state.
                        for r_idx in range(len(b.batch)):
                            key = (sample_ids[r_idx], int(rollout_idxs[r_idx]))
                            latest_seen[key] = (b_idx, r_idx)
                            
                    # print(f"[STEP {self.global_steps}] Found {len(latest_seen)} unique conversations (terminal states).")
                    
                    # --- 1.15 INNER SAMPLING BATCH EXECUTION ---
                    # Always executed exactly once per step (we only reach this point at the final chunk).
                    # inner_sampling_buffer accumulates across all sessions; flush it here.
                    total_inner_states = sum(len(b.batch) for b in inner_sampling_buffer)

                    if total_inner_states > 0:
                        # print(f"[STEP {self.global_steps}] Executing Inner Sampling Batch...")
                        inner_batch_full = DataProto.concat(inner_sampling_buffer)
                        # DataProto.concat keeps the first meta_info by reference.
                        # Isolate before any in-place union/meta mutations in the inner branch.
                        inner_batch_full.meta_info = deepcopy(inner_batch_full.meta_info)
                        inner_sampling_buffer.clear()
                        
                        # Repeat inner samples mapped to the full context
                        inner_batch_full = inner_batch_full.repeat(repeat_times=inner_n, interleave=True)
                        n_inner_rollouts = inner_n
                        inner_rollout_idx = torch.arange(0, n_inner_rollouts).repeat(len(inner_batch_full.batch) // n_inner_rollouts)
                        inner_batch_full.batch['rollout_idx'] = inner_rollout_idx.numpy()

                        # Create the subset generator batch exactly like standard gen_batch
                        inner_gen_batch = inner_batch_full.select(
                            batch_keys=['rollout_idx', 'source_rollout_idx', 'batch_idx', 'epoch'], 
                            non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json', 'cumulative_session_tokens'], 
                            meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'],
                            deepcopy=True
                        )

                        # DataProto often shares meta_info references across views/concats;
                        # isolate inner branch mutations from normal rollout state.
                        inner_gen_batch.meta_info = deepcopy(inner_gen_batch.meta_info)

                        # Set standard validation generation params
                        inner_gen_batch.meta_info.update({
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample, # Use validation kwargs for inner sampling
                            'validate': False, # Treat as training to keep graph attached later
                            'memory_snapshot_suffix': 'inner',
                        })

                        # Pad & Generate
                        inner_gen_batch_padded, pad_size = pad_dataproto_to_divisor(inner_gen_batch, self.actor_rollout_wg.world_size)
                        with _timer('gen_inner', timing_raw):
                            inner_output_padded = self.actor_rollout_wg.multi_turn_generate_sequences(inner_gen_batch_padded)
                        inner_output = unpad_dataproto(inner_output_padded, pad_size=pad_size)
                        inner_output.meta_info = deepcopy(inner_output.meta_info)

                        # Pop generation-specific keys from both metainfos to prevent union conflicts
                        for k in ['eos_token_id', 'pad_token_id', 'recompute_log_prob', 'do_sample', 'validate']:
                            inner_gen_batch.meta_info.pop(k, None)
                            inner_output.meta_info.pop(k, None)

                        # Union output with FULL prompt data
                        inner_eval_batch = inner_batch_full.union(inner_output)
                        inner_eval_batch.meta_info = deepcopy(inner_eval_batch.meta_info)
                        
                        # Evaluate QA Reward on inner_eval_batch
                        inner_eval_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                        inner_eval_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                        # Inner reward must read/write the isolated inner namespace only.
                        inner_eval_batch.meta_info['memory_snapshot_suffix'] = 'inner'
                        with _timer('reward_inner', timing_raw):
                            # This evaluates the turn immediately
                            reward_tensor_map_inner = self.reward_fn(inner_eval_batch, compression_penalty=self.config.trainer.compression_penalty)
                        
                        # Assign turn-level reward strictly per role (no cross-role fallback).
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        max_turns = self.config.actor_rollout_ref.rollout.max_num_turns
                        bsz = len(inner_eval_batch.batch)

                        curr_num_turns = inner_eval_batch.non_tensor_batch['num_turns']
                        
                        agent_roles = inner_eval_batch.meta_info.get('agent_roles', ['meta_thinking', 'reasoning'])
                        
                        for role in agent_roles:
                            role_reward_key = f'{role}_turn_level_reward'
                            assert role_reward_key in reward_tensor_map_inner, (
                                f"Inner reward integrity failure: missing '{role_reward_key}' in reward tensor map."
                            )
                            inner_rewards_all = reward_tensor_map_inner[role_reward_key].sum(dim=-1).to(device)  # (B,)
                            propagated_reward = torch.zeros((bsz, max_turns), device=device)
                            
                            for r_idx in range(bsz):
                                last_turn_idx = curr_num_turns[r_idx] - 1
                                if 0 <= last_turn_idx < max_turns:
                                    propagated_reward[r_idx, last_turn_idx] = inner_rewards_all[r_idx]

                            inner_eval_batch.batch[role_reward_key] = propagated_reward
                            turn_mask = verl_F.get_turn_mask(propagated_reward, curr_num_turns)
                            key_return = role_reward_key.replace('reward', 'return')
                            inner_eval_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                propagated_reward, turn_mask, self.config.algorithm.gamma_turn_level)

                        # Keep TensorDict schema aligned with normal batches before final concat.
                        if 'source_rollout_idx' in inner_eval_batch.batch.keys():
                            inner_eval_batch.batch.pop('source_rollout_idx')

                        evaluated_inner_batches.append(inner_eval_batch)

                    # 1.2 Group and Extract Terminal States
                    # We need to construct `terminal_batch` containing these final states.
                    # We also need `terminal_keys_ordered` to map the resulting rows back to their IDs for reward assignment.
                    
                    # Group locations by batch index to minimize slicing operations and ensure deterministic order.
                    # b_idx -> list of (row_idx, key)
                    b_idx_to_keys = defaultdict(list)
                    for key, (b_idx, r_idx) in latest_seen.items():
                        # [batch_idx] -> list of [row_idx in batch, (sample_id, rollout_idx)]
                        b_idx_to_keys[b_idx].append((r_idx, key))
                    
                    terminal_sub_batches = []
                    terminal_keys_ordered = [] # Will store (sample_id, rollout_idx) for each row in the final terminal_batch
                    
                    # Iterate through batches in order (Batch 0, Batch 1, ...)
                    for b_idx in sorted(b_idx_to_keys.keys()):
                        # Sort by row index within the batch. 
                        # This preserves the relative order of samples as they appeared in the original batch
                        # and allows for efficient slicing (if indices were contiguous, though here we use list indexing).
                        rows_and_keys = sorted(b_idx_to_keys[b_idx], key=lambda x: x[0])
                        row_indices = [x[0] for x in rows_and_keys]
                        keys = [x[1] for x in rows_and_keys]
                        
                        # Extract the terminal rows from this specific batch
                        sub_batch = all_batches[b_idx][row_indices]
                        # Since we sorted by batch index and then row index, we can just append
                        # terminal_sub_batches is now a list of DataProto batches in the order they should be processed
                        terminal_sub_batches.append(sub_batch)
                        
                        # Record the keys corresponding to these rows, keys are (sample_id, rollout_idx)
                        terminal_keys_ordered.extend(keys)
                        
                    terminal_batch = DataProto.concat(terminal_sub_batches)
                    terminal_batch.meta_info = deepcopy(terminal_batch.meta_info)
                    # print(f"[STEP {self.global_steps}] Terminal batch constructed with size {len(terminal_batch.batch)}")


                    # --- 2. Compute Rewards on Terminal Batch ---
                    with _timer('reward', timing_raw):
                        # print(f"\n[STEP {self.global_steps}] Computing rewards for Terminal Batch...")
                        
                        if self.use_rm:
                             raise NotImplementedError('RM is not implemented for delayed feedback yet')

                        terminal_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                        terminal_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                        # Terminal reward must always use the normal snapshot namespace.
                        terminal_batch.meta_info['memory_snapshot_suffix'] = ''
                        
                        reward_tensor_map = self.reward_fn(terminal_batch, compression_penalty=self.config.trainer.compression_penalty)
                        # print(f"[STEP {self.global_steps}] Reward computed. Keys: {list(reward_tensor_map.keys())}")
                        
                        # Process metrics (logging only)
                        category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial']
                        for cat_name in category_names:
                            f1_sum_key = f'{cat_name}_f1_sum'
                            if f1_sum_key in reward_tensor_map:
                                f1_sum = reward_tensor_map.pop(f1_sum_key).item()
                                bleu_sum = reward_tensor_map.pop(f'{cat_name}_bleu_sum').item()
                                count = int(reward_tensor_map.pop(f'{cat_name}_count').item())
                                if count > 0:
                                    metrics[f'train/{cat_name}_f1'] = f1_sum / count
                                    metrics[f'train/{cat_name}_bleu'] = bleu_sum / count
                                    metrics[f'train/{cat_name}_count'] = count
                        
                        # Keep per-session tensors for propagation below; only pop true metric-only keys.
                        keys_to_pop = [
                            key for key in reward_tensor_map.keys()
                            if not key.endswith('_turn_level_reward')
                            and key not in ('per_session_f1', 'cumulative_per_session_f1')
                        ]
                        scalar_metric_keys = [
                            'memory_size', 'memory_insert_count', 'memory_delete_count', 'memory_token_count', 'memory_compression_ratio',
                            'memory_update_count', 'memory_ops', 'evidence_precision', 
                            'evidence_recall', 'avg_retrieval_rank', 'memory_failure_rate',
                            'retrieval_failure_rate', 'total_failure_rate'
                        ]
                        
                        for key in keys_to_pop:
                            tensor_value = reward_tensor_map.pop(key)
                            if key in scalar_metric_keys or tensor_value.dim() == 0:
                                metric_key = f'memory/{key}' if key in scalar_metric_keys else f'train/{key}'
                                metrics[metric_key] = tensor_value.item() if tensor_value.dim() == 0 else tensor_value
                            else:
                                mean_val = tensor_value.float().mean().item()
                                metrics[f'train/{key}'] = mean_val
                                if key in ['acc', 'bleu', 'evidence']:
                                    metrics[f'critic/{key}'] = mean_val

                        if 'per_session_f1' in reward_tensor_map:
                            metrics['train/per_session_f1'] = reward_tensor_map['per_session_f1'].float().mean().item()
                        if 'cumulative_per_session_f1' in reward_tensor_map:
                            metrics['train/cumulative_per_session_f1'] = reward_tensor_map['cumulative_per_session_f1'].float().mean().item()
                        
                        # --- 3. Build Global Reward Maps ---
                        # Build one strict outcome map per role.
                        global_reward_map_by_role = {} # role -> {(sample_id, rollout_idx): reward_value (float)}
                        global_per_session_map = {} # (sample_id, rollout_idx) -> per_session_row (Tensor)
                        
                        agent_roles = terminal_batch.meta_info['agent_roles']

                        for role in agent_roles:
                            role_reward_key = f'{role}_turn_level_reward'
                            assert role_reward_key in reward_tensor_map, (
                                f"Reward integrity failure: missing '{role_reward_key}' in terminal reward tensor map."
                            )
                            rewards_all = reward_tensor_map[role_reward_key] # (B_term, T)
                            outcome_rewards_all = rewards_all.sum(dim=-1) # (B_term,)

                            role_map = {}
                            for i, key in enumerate(terminal_keys_ordered):
                                role_map[key] = outcome_rewards_all[i].item()
                            global_reward_map_by_role[role] = role_map


                        reward_type = self.config.trainer.get('rewardtype', 'global')
                        # print(f"[STEP {self.global_steps}] Reward type from config: {reward_type}")

                        if reward_type == 'cumulative' or reward_type == 'cumulative_per_session_f1':
                            if 'cumulative_per_session_f1' in reward_tensor_map:
                                # print(f"[STEP {self.global_steps}] Using cumulative per-session F1 for global session map.")
                                per_session_f1_all = reward_tensor_map['cumulative_per_session_f1'] # (B_term, MaxSessions)
                                for i, key in enumerate(terminal_keys_ordered):
                                    global_per_session_map[key] = per_session_f1_all[i]
                                # print(f"[STEP {self.global_steps}] Global per-session reward map built (cumulative).")
                            # else:
                                # print(f"[WARNING] Cumulative reward requested (rewardtype={reward_type}) but 'cumulative_per_session_f1' not found in map!")
                        
                        elif reward_type == 'persession' or reward_type == 'per_session_f1':
                            if 'per_session_f1' in reward_tensor_map:
                                per_session_f1_all = reward_tensor_map['per_session_f1'] # (B_term, MaxSessions)
                                for i, key in enumerate(terminal_keys_ordered):
                                    global_per_session_map[key] = per_session_f1_all[i]
                                # print(f"[STEP {self.global_steps}] Global per-session reward map built.")
                            # else:
                                # print(f"[WARNING] Per-session reward requested (rewardtype={reward_type}) but 'per_session_f1' not found in map!")
                                
                        # elif reward_type == 'global':
                            # print(f"[STEP {self.global_steps}] Using global outcome reward. Bypassing per-session map.")
                        # else:
                            # print(f"[STEP {self.global_steps}] Unrecognized reward_type '{reward_type}'. Defaulting to global outcome reward.")


                        # --- 4. Propagate Rewards to All Batches ---
                        # print(f"[STEP {self.global_steps}] Propagating rewards to all {len(all_batches)} batches...")
                        
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        max_turns = self.config.actor_rollout_ref.rollout.max_num_turns
                        
                        per_session_hits = 0
                        invalid_session_hits = 0
                        op_shaping_session_hits = 0

                        insert_penalty = float(self.config.trainer.get('insert_penalty', 0.0))
                        update_bonus = float(self.config.trainer.get('update_bonus', 0.0))
                        delete_bonus = float(self.config.trainer.get('delete_bonus', 0.0))

                        def _extract_scalar(value, field_name: str):
                            if isinstance(value, torch.Tensor):
                                if value.numel() != 1:
                                    raise TypeError(
                                        f"Expected scalar for {field_name}, got tensor with shape={tuple(value.shape)}"
                                    )
                                return value.item()
                            if isinstance(value, np.ndarray):
                                if value.size != 1:
                                    raise TypeError(
                                        f"Expected scalar for {field_name}, got ndarray with shape={value.shape}"
                                    )
                                return value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], 'item') else value.reshape(-1)[0]
                            if isinstance(value, (list, tuple)):
                                if len(value) != 1:
                                    raise TypeError(
                                        f"Expected scalar for {field_name}, got sequence with len={len(value)}"
                                    )
                                return _extract_scalar(value[0], field_name)
                            return value

                        for b_idx, target_batch in enumerate(all_batches):
                            # info for this batch
                            curr_sample_ids = target_batch.non_tensor_batch['sample_id']
                            curr_rollout_idxs = target_batch.batch['rollout_idx']
                            if isinstance(curr_rollout_idxs, torch.Tensor):
                                curr_rollout_idxs = curr_rollout_idxs.cpu().tolist()
                            curr_num_turns = target_batch.non_tensor_batch['num_turns']

                            # Determine session index per sample (chunk_id is 1-based: Session 1, Session 2...)
                            chunk_ids_raw = target_batch.non_tensor_batch['chunk_id']
                            if isinstance(chunk_ids_raw, torch.Tensor):
                                chunk_ids = chunk_ids_raw.cpu().tolist()
                            elif isinstance(chunk_ids_raw, np.ndarray):
                                chunk_ids = chunk_ids_raw.tolist()
                            elif isinstance(chunk_ids_raw, (list, tuple)):
                                chunk_ids = list(chunk_ids_raw)
                            else:
                                chunk_ids = [chunk_ids_raw] * len(curr_sample_ids)
                            if not isinstance(chunk_ids, list):
                                chunk_ids = [chunk_ids] * len(curr_sample_ids)
                            if len(chunk_ids) != len(curr_sample_ids):
                                chunk_ids = [chunk_ids[0]] * len(curr_sample_ids)

                            def _normalize_to_list(raw_value, expected_len: int):
                                if isinstance(raw_value, torch.Tensor):
                                    values = raw_value.cpu().tolist()
                                elif isinstance(raw_value, np.ndarray):
                                    values = raw_value.tolist()
                                elif isinstance(raw_value, (list, tuple)):
                                    values = list(raw_value)
                                else:
                                    values = [raw_value] * expected_len

                                if not isinstance(values, list):
                                    values = [values] * expected_len
                                if len(values) != expected_len:
                                    values = [values[0]] * expected_len
                                return values

                            insert_counts = _normalize_to_list(
                                target_batch.non_tensor_batch.get('mem_insert_successful', 0), len(curr_sample_ids)
                            )
                            delete_counts = _normalize_to_list(
                                target_batch.non_tensor_batch.get('mem_delete_successful', 0), len(curr_sample_ids)
                            )
                            update_counts = _normalize_to_list(
                                target_batch.non_tensor_batch.get('mem_update_successful', 0), len(curr_sample_ids)
                            )

                            bsz = len(target_batch.batch)

                            # For each role, apply rewards
                            for role in agent_roles:
                                role_reward_key = f'{role}_turn_level_reward'
                                role_global_reward_map = global_reward_map_by_role.get(role)
                                assert role_global_reward_map is not None, (
                                    f"Reward propagation integrity failure: missing global reward map for role '{role}'."
                                )
                                propagated_reward = torch.zeros((bsz, max_turns), device=device)
                                
                                for r_idx in range(bsz):
                                    rollout_idx_scalar = _extract_scalar(curr_rollout_idxs[r_idx], 'rollout_idx')
                                    key = (curr_sample_ids[r_idx], int(rollout_idx_scalar))
                                    sess_idx_1based = _extract_scalar(chunk_ids[r_idx], 'chunk_id')
                                    sess_idx_0based = int(sess_idx_1based) - 1
                                    
                                    if key not in role_global_reward_map:
                                        raise AssertionError(
                                            f"Reward propagation integrity failure: missing global reward for role={role}, key={key}."
                                        )

                                    # Strategy A: Use per-session F1 if available (Dense Reward)
                                    if key in global_per_session_map:
                                        per_session_row = global_per_session_map[key]
                                        if not (0 <= sess_idx_0based < per_session_row.shape[0]):
                                            invalid_session_hits += 1
                                            raise AssertionError(
                                                f"Reward propagation integrity failure: invalid session index {sess_idx_0based} "
                                                f"for key={key} with per_session length={per_session_row.shape[0]}."
                                            )
                                        reward_val = per_session_row[sess_idx_0based].item()
                                        if update_bonus > 0.0 or delete_bonus > 0.0:
                                            inserts = float(_extract_scalar(insert_counts[r_idx], 'mem_insert_successful'))
                                            deletes = float(_extract_scalar(delete_counts[r_idx], 'mem_delete_successful'))
                                            updates = float(_extract_scalar(update_counts[r_idx], 'mem_update_successful'))
                                            # Ratio-based shaping: reward the *proportion* of ops that are update/delete.
                                            # This prevents reward hacking (spamming ops to inflate raw-count bonuses)
                                            # and avoids destabilizing the QA signal with an unbounded insert penalty.
                                            total_ops = inserts + updates + deletes
                                            if total_ops > 0:
                                                update_ratio = updates / total_ops
                                                delete_ratio = deletes / total_ops
                                                op_bonus = (update_bonus * update_ratio) + (delete_bonus * delete_ratio)
                                                reward_val += min(op_bonus, 0.1)
                                            op_shaping_session_hits += 1
                                        per_session_hits += 1
                                    else:
                                        # Strategy B: strict outcome reward (no fallback chains)
                                        reward_val = float(role_global_reward_map[key])
                                            
                                    # Assign to the last turn of this session
                                    last_turn_idx = curr_num_turns[r_idx] - 1
                                    if last_turn_idx >= 0 and last_turn_idx < max_turns:
                                        propagated_reward[r_idx, last_turn_idx] = reward_val

                                target_batch.batch[role_reward_key] = propagated_reward
                                
                                # Compute Turn Level Return (GAE/Discounting happens here)
                                turn_mask = verl_F.get_turn_mask(propagated_reward, curr_num_turns)
                                key_return = role_reward_key.replace('reward', 'return')
                                target_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                    propagated_reward, turn_mask, self.config.algorithm.gamma_turn_level)

                        metrics['train/reward_per_session_hits'] = float(per_session_hits)
                        metrics['train/reward_invalid_session_hits'] = float(invalid_session_hits)
                        metrics['train/reward_op_shaping_session_hits'] = float(op_shaping_session_hits)

                        assert invalid_session_hits == 0, (
                            f"Reward propagation integrity failure: invalid_session_hits={invalid_session_hits}. "
                            "At least one sample had session index outside per-session reward bounds."
                        )

                        # Concatenate everything for PPO update
                        # print(f"[STEP {self.global_steps}] Concatenating {len(all_batches)} batches for update...")
                        batch = DataProto.concat(all_batches + evaluated_inner_batches)
                        
                        # Fix for DP chunking error: Ensure the final batch size (after role expansion) is perfectly 
                        # divisible by world_size by dropping dangling inner samples at the very end of the batch.
                        target_bsz_before_roles = len(batch.batch)
                        while (target_bsz_before_roles * len(agent_roles)) % self.actor_rollout_wg.world_size != 0:
                            target_bsz_before_roles -= 1
                            
                        if target_bsz_before_roles < len(batch.batch):
                            import copy
                            # slice dataproto to truncate
                            batch = batch[:target_bsz_before_roles]
                            # we must manually re-copy the meta info as slicing loses or proxies it sometimes
                            batch.meta_info = copy.deepcopy(batch.meta_info)

                        self.accumulated_batches = [] # Clear accumulation
                        evaluated_inner_batches = []
                        # print(f"[STEP {self.global_steps}] Mega-batch size: {len(batch.batch)}")

                    with _timer('adv', timing_raw):
                        # print(f"\n[STEP {self.global_steps}] Computing advantages (Trajectory Aggregated)...")
                        # Merge different role data into a single DataProto
                        merged_batch = merge_roles_data(batch)

                        # Recalculate global_token_num for the entire merged mega-batch
                        # This ensures the metadata has the correct length (2 * MegaBatchSize) 
                        # and accurately reflects all tokens across all sessions and inner samples.
                        merged_batch.meta_info['global_token_num'] = (
                            merged_batch.batch['attention_mask'].view(len(merged_batch.batch), -1).sum(-1)
                        ).tolist()

                        # recompute old_log_probs (on merged data)
                        with _timer('old_log_prob', timing_raw):
                            # print(f"\n[STEP {self.global_steps}] Computing old log probabilities (merged)...")
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(merged_batch)
                            merged_batch = merged_batch.union(old_log_prob)
                            # print(f"[STEP {self.global_steps}] Old log probs computed.")

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                # print(f"[STEP {self.global_steps}] Computing reference log probabilities (merged)...")
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(merged_batch)
                                merged_batch = merged_batch.union(ref_log_prob)
                                # print(f"[STEP {self.global_steps}] Reference log probs computed.")
                        
                        # assign turn_level scores to the last token of each turn
                        token_level_scores = compute_token_level_scores(merged_batch)
                        merged_batch.batch['token_level_scores'] = token_level_scores
                        batch = merged_batch
                        
                        # Apply KL penalty to token-level rewards if enabled
                        # This makes rewards dense (per-token KL signal) instead of sparse (turn-end only)
                        if self.config.algorithm.get('use_kl_in_reward', False) and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute critic values for GAE (after merge, before advantage computation)
                        if self.use_critic:
                            with _timer('values', timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)
                            
                            # Compute cross-session bootstrap values
                            # For each non-terminal session, bootstrap from the next session's
                            # first turn critic value instead of treating the boundary as terminal
                            gamma_session = self.config.algorithm.get('gamma_session_level', 1.0)
                            bootstrap_vals = compute_bootstrap_values(batch, gamma_session=gamma_session)
                            batch.batch['bootstrap_value'] = bootstrap_vals
                            n_nonzero = (bootstrap_vals != 0).sum().item()
                            print(f"[STEP {self.global_steps}] Cross-session bootstrap (gamma_session={gamma_session}): "
                                  f"{n_nonzero} non-zero / {len(bootstrap_vals)} total rows "
                                  f"(mean of non-zero={bootstrap_vals[bootstrap_vals != 0].mean():.4f})" if n_nonzero > 0 
                                  else f"[STEP {self.global_steps}] Cross-session bootstrap: no bootstrapping (single session or no critic)")

                        # print(f"[STEP {self.global_steps}] Computing advantages with {self.config.algorithm.adv_estimator}...")
                        batch.meta_info['gamma_turn_level'] = self.config.algorithm.gamma_turn_level
                        batch.meta_info['use_bilevel_gae'] = self.config.algorithm.get('use_bilevel_gae', False)
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma_token_level,
                                                  lam=self.config.algorithm.lam_token_level,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        # print(f"[STEP {self.global_steps}] Advantages computed.")


                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            # print(f"\n[STEP {self.global_steps}] Updating critic...")
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        # print(f"[STEP {self.global_steps}] Critic updated.")

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            # print(f"\n[STEP {self.global_steps}] Updating actor...")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        # print(f"[STEP {self.global_steps}] Actor updated.")
                    # else:
                        # print(f"\n[STEP {self.global_steps}] Skipping actor update (warmup: {self.global_steps}/{self.config.trainer.critic_warmup})")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            # print(f"\n[STEP {self.global_steps}] Running validation...")
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            # print(f"[STEP {self.global_steps}] Validation complete.")
                        
                        # --- Update Best Checkpoint Info ---
                        current_val_acc = val_metrics.get('val/acc/locomo', -1.0)
                        if current_val_acc > self.best_val_acc:
                            # print(f"[STEP {self.global_steps}] New best validation accuracy: {current_val_acc:.4f} (was {self.best_val_acc:.4f})")
                            self.best_val_acc = current_val_acc
                            self.best_global_step = self.global_steps
                            self.patience_counter = 0
                            
                            # Save best info to file
                            best_info_path = os.path.join(self.config.trainer.default_local_dir, 'best_checkpoint_info.txt')
                            try:
                                with open(best_info_path, 'w') as f:
                                    f.write(f"{self.best_global_step}")
                                # print(f"[STEP {self.global_steps}] Saved best checkpoint info to {best_info_path}")
                            except Exception as e:
                                print(f"[WARN] Failed to save best checkpoint info: {e}")
                        else:
                            self.patience_counter += 1
                                
                        metrics.update(val_metrics)

                        max_patience = self.config.trainer.get('early_stop_patience', 3)
                        if max_patience > 0 and self.patience_counter >= max_patience:
                            print(f"\n[STEP {self.global_steps}] Early stopping triggered! Validation accuracy hasn't improved for {self.patience_counter} evaluations. Best acc: {self.best_val_acc:.4f} at step {self.best_global_step}.")
                            is_last_step = True

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            # print(f"\n[STEP {self.global_steps}] Saving checkpoint...")
                            self._save_checkpoint()
                            # print(f"[STEP {self.global_steps}] Checkpoint saved.")

                # collect metrics
                # print(f"\n[STEP {self.global_steps}] Computing and logging metrics...")
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                # print(f"[STEP {self.global_steps}] Metrics logged. Step complete.\n")

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                all_negative_cnt = 0
                all_positive_cnt = 0
                kept_prompt_cnt = 0
                total_prompt_cnt = 0

                if is_last_step:
                    # run test evaluation at end of training
                    if self.test_dataloader is not None and self.config.trainer.get('test_after_train', True):
                        test_metrics = self._test()
                        pprint(f'Final test metrics: {test_metrics}')
                        logger.log(data=test_metrics, step=self.global_steps)
                    del logger
                    try:
                        # flush wandb pending logs
                        import wandb
                        if wandb.run is not None:
                            wandb.finish()
                    except ImportError:
                        pass
                    return

                self.global_steps += 1
                # print(f"[FIT] Incrementing global_steps to {self.global_steps}")

    def _save_train_generations(self, batch: DataProto):
        # save train generations
        output_dir = Path(self.config.trainer.default_local_dir) / 'replay_buffer'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'train_step_{self.global_steps}.jsonl'
        
        
        results_dict = {}
        for i, data_item in enumerate(batch):
            uid = data_item.non_tensor_batch['uid']
            if uid not in results_dict:
                qa_pairs = json.loads(data_item.non_tensor_batch['qa_pairs_json'])
                results_dict[uid] = {
                    "question": data_item.non_tensor_batch['question'],
                    # TODO:: handle multi-questions case
                    "groundtruth": qa_pairs[0]['answer'] if qa_pairs else "N/A",
                    "response": [],
                    "history": [],
                    "score": [],
                    "finish_reason": [],
                }

            padded_history = data_item.non_tensor_batch['history']
            unpad_history = [x for x in padded_history if x['role'] != 'padding']
            results_dict[uid]['history'].append(unpad_history)
            results_dict[uid]['response'].append(data_item.non_tensor_batch['response'])
            results_dict[uid]['score'].append(
                data_item.batch['reasoning_turn_level_reward'].sum().item()
            )
            results_dict[uid]['finish_reason'].append(
                data_item.non_tensor_batch['finish_reason']
            )

        results_to_save = []
        for uid, result in results_dict.items():
            result['avg_score'] = sum(result['score']) / len(result['score'])
            results_to_save.append(result)
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(results_to_save)    