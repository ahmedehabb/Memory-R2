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
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
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


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        raise NotImplementedError('GAE is not implemented yet')
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
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
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        
        # since we will train two agents data
        self.config.actor_rollout_ref.actor.ppo_mini_batch_size *= 2

        self._create_dataloader()

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
                pad_incomplete=True  # Pad training batches with repeats (acts like extra rollouts)
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

        # Create test dataset and dataloader if test_only mode is enabled
        if self.config.trainer.get('test_only', False):
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
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

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
        print("\n" + "="*80)
        print("STARTING VALIDATION")
        print("="*80)
        
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
        print(f"\n[VALIDATE] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None, # changed this to None from FINISH_FLAG
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
            print(f"[VALIDATE] Multi-turn mode enabled with FINISH_FLAG")
            print(f"[VALIDATE] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[VALIDATE] agent_roles: {rollout_meta_info['agent_roles']}")
            print(f"[VALIDATE] finish_flag: {rollout_meta_info['finish_flag'][:50] if rollout_meta_info['finish_flag'] else None}...")
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
            print(f"[VALIDATE] Single-turn mode enabled (no FINISH_FLAG)")
            print(f"[VALIDATE] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[VALIDATE] agent_roles: {rollout_meta_info['agent_roles']}")

        print(f"\n[VALIDATE] Starting validation loop: {len(self.val_dataloader)} batches")
        print(f"[VALIDATE] Strategy: Check qa_pairs_json for each sample - if non-empty, conversation has ended and will be evaluated")
        total_batches = len(self.val_dataloader)
        
        for batch_idx, test_data in enumerate(self.val_dataloader):
            print(f"\n{'*'*80}")
            print(f"VALIDATION BATCH {batch_idx + 1}/{total_batches}")
            print(f"{'*'*80}")
            print(f"\n[VAL BATCH {batch_idx + 1}] Creating batch from dataloader...")
            print(f"[VAL BATCH {batch_idx + 1}] Batch size: {len(test_data['question'])}")
            print(f"[VAL BATCH {batch_idx + 1}] test_data keys: {list(test_data.keys())}")
            
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_data['epoch'] = torch.full((len(test_data['question']),), self.global_steps, dtype=torch.long)
            
            # Add validation epoch/split info
            # rollout_meta_info['epoch'] = self.global_steps  # validation epoch
            rollout_meta_info['split'] = 'validation'
            
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)
            print(f"[VAL BATCH {batch_idx + 1}] test_batch.batch keys: {list(test_batch.batch.keys())}")
            print(f"[VAL BATCH {batch_idx + 1}] test_batch.non_tensor_batch keys: {list(test_batch.non_tensor_batch.keys())}")

            # Check which samples have finished (non-zero num_questions) BEFORE repeating
            num_questions_list = test_batch.non_tensor_batch['num_qas']
            finished_mask = [num_questions > 0 for num_questions in num_questions_list]
            num_finished = sum(finished_mask)
            print(f"[VAL BATCH {batch_idx + 1}] Found {num_finished}/{len(finished_mask)} finished conversations (with non-empty qa_pairs_json)")

            # Store original inputs and ground truths BEFORE repeating
            input_texts = test_batch.non_tensor_batch['question']
            sample_inputs.extend(input_texts)
            print(f"[VAL BATCH {batch_idx + 1}] Collected {len(input_texts)} input texts")

            # Store original ground truth if available
            # TODO: support multi- groundtruths
            ground_truths = [json.loads(x)[0]["answer"] if json.loads(x) else "N/A" for x in test_batch.non_tensor_batch['qa_pairs_json']]
            sample_groundtruths.extend(ground_truths)
            print(f"[VAL BATCH {batch_idx + 1}] Collected {len(ground_truths)} ground truths")

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)
            print(f"[VAL BATCH {batch_idx + 1}] Repeated batch {self.config.actor_rollout_ref.rollout.val_kwargs.n} times. New size: {len(test_batch.batch)}")
            
            # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
            rollout_idx = torch.arange(0, len(test_batch.batch))
            test_batch.batch['rollout_idx'] = rollout_idx
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                print(f"[VAL BATCH {batch_idx + 1}] Skipping model-based reward model validation")
                return {}

            print(f"\n[VAL BATCH {batch_idx + 1}] Preparing generation batch...")
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('multi_modal_inputs validation not implemented yet')
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['rollout_idx', 'batch_idx', 'epoch'], 
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'], 
                        deepcopy=True
                    )
            
            print(f"[VAL BATCH {batch_idx + 1}] Generation batch prepared with {len(test_gen_batch.batch)} samples")
            print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.batch keys: {list(test_gen_batch.batch.keys())}")
            print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.non_tensor_batch keys: {list(test_gen_batch.non_tensor_batch.keys())}")
            print(f"[VAL BATCH {batch_idx + 1}] test_gen_batch.meta_info keys: {list(test_gen_batch.meta_info.keys())}")
            
            test_gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            print(f'[VAL BATCH {batch_idx + 1}] test_gen_batch meta_info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            print(f"\n[VAL BATCH {batch_idx + 1}] Padding to be divisible by world_size={self.actor_rollout_wg.world_size}...")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            print(f"[VAL BATCH {batch_idx + 1}] Padded batch size: {len(test_gen_batch_padded.batch)}, pad_size: {pad_size}")
            
            print(f"[VAL BATCH {batch_idx + 1}] >>> Calling multi_turn_generate_sequences...")
            test_output_gen_batch_padded = self.actor_rollout_wg.multi_turn_generate_sequences(test_gen_batch_padded)
            print(f"[VAL BATCH {batch_idx + 1}] <<< Generation complete")

            # unpad
            print(f"[VAL BATCH {batch_idx + 1}] Unpadding batch...")
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(f'[VAL BATCH {batch_idx + 1}] Validation generation end. Output batch size: {len(test_output_gen_batch.batch)}')

            # Store generated outputs
            print(f"\n[VAL BATCH {batch_idx + 1}] Processing outputs...")
            output_texts = test_output_gen_batch.non_tensor_batch['response']
            sample_outputs.extend(output_texts)
            print(f"[VAL BATCH {batch_idx + 1}] Collected {len(output_texts)} output texts")
            print(f"[VAL BATCH {batch_idx + 1}] Sample output[0]: {output_texts[0][:100] if isinstance(output_texts[0], str) else output_texts[0]}...")

            history_lst.append(test_output_gen_batch.non_tensor_batch['history'].tolist())
            
            # Collect generation metrics (num_turns, completion_tokens) from ALL batches
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            print(f"[VAL BATCH {batch_idx + 1}] num_turns (all samples): min={num_turns.min()}, max={num_turns.max()}, mean={num_turns.float().mean()}")
            
            turn_level_completion_tokens = test_output_gen_batch.batch['meta_thinking_num_gen_tokens'].cpu() + \
                test_output_gen_batch.batch['reasoning_num_gen_tokens'].cpu()
            completion_tokens = turn_level_completion_tokens.sum(dim=-1)
            completion_tokens_lst.append(completion_tokens)
            print(f"[VAL BATCH {batch_idx + 1}] completion_tokens (all samples): min={completion_tokens.min()}, max={completion_tokens.max()}, mean={completion_tokens.float().mean()}")
            
            # Check if any conversations finished in this batch and evaluate them
            if num_finished > 0:
                print(f"[VAL BATCH {batch_idx + 1}] Evaluating {num_finished} finished conversations...")
                print(f"[VAL BATCH {batch_idx + 1}] Merging generation output with test batch...")
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
                
                print(f"[VAL BATCH {batch_idx + 1}] Filtering to {len(finished_indices)} samples (finished conversations after repeating)")
                
                # Select only finished samples
                finished_test_batch = test_batch_with_gen[finished_indices]
                
                # Compute rewards for finished conversations
                print(f"[VAL BATCH {batch_idx + 1}] Computing rewards for finished conversations...")
                reward_tensor = self.val_reward_fn(finished_test_batch)
                print(f"[VAL BATCH {batch_idx + 1}] Reward tensor keys: {list(reward_tensor.keys())}")
                
                reward_tensor_lst.append(reward_tensor['reasoning_turn_level_reward'])
                reward_tensor_dict_lst.append(reward_tensor)  # Store full dict
                acc_tensor_lst.append(reward_tensor['acc'])
                bleu_tensor_lst.append(reward_tensor['bleu'])
                print(f"[VAL BATCH {batch_idx + 1}] reasoning_turn_level_reward shape: {reward_tensor['reasoning_turn_level_reward'].shape}")
                print(f"[VAL BATCH {batch_idx + 1}] acc shape: {reward_tensor['acc'].shape}")
                print(f"[VAL BATCH {batch_idx + 1}] bleu shape: {reward_tensor['bleu'].shape}")
                
                # Store scores
                scores = reward_tensor['reasoning_turn_level_reward'].sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                print(f"[VAL BATCH {batch_idx + 1}] Sample scores[0]: {scores[0]}")
                
                # Get data sources from finished samples
                data_source_lst.append(finished_test_batch.non_tensor_batch.get('subset', ['locomo'] * reward_tensor['reasoning_turn_level_reward'].shape[0]))
            else:
                print(f"[VAL BATCH {batch_idx + 1}] No finished conversations in this batch, skipping reward computation")
            
            print(f"[VAL BATCH {batch_idx + 1}] Batch processing complete\n")

        # Now compute final metrics from all finished conversations across all batches
        print(f"\n[VALIDATE] All batches processed. Computing final metrics...")
        
        if len(reward_tensor_lst) == 0:
            print("[VALIDATE] WARNING: No finished conversations found across all batches!")
            return {}
        
        # Log generations
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, groundtruths=sample_groundtruths, histories=history_lst)

        # Concatenate all accumulated tensors
        print(f"[VALIDATE] Concatenating reward tensors from {len(reward_tensor_lst)} batches with finished conversations...")
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (total_finished_samples,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        bleu_tensor = torch.cat(bleu_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        print(f"[VALIDATE] Total finished samples evaluated: {reward_tensor.shape[0]}")
        print(f"[VALIDATE] Mean reward: {reward_tensor.mean().item():.4f}")
        print(f"[VALIDATE] Mean accuracy: {acc_tensor.mean().item():.4f}")
        print(f"[VALIDATE] Mean BLEU: {bleu_tensor.mean().item():.4f}")

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
            print(f"[VALIDATE] {data_source} mean reward: {np.mean(rewards):.4f}")
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)
            print(f"[VALIDATE] {data_source} mean accuracy: {np.mean(accs):.4f}")
        for data_source, bleus in data_source_bleu.items():
            metric_dict[f'val/bleu/{data_source}'] = np.mean(bleus)
            print(f"[VALIDATE] {data_source} mean BLEU: {np.mean(bleus):.4f}")
        
        # Stage 2 aggregation: Combine per-category metrics across ALL validation batches
        # Each batch returns sum and count (not averages), so we just accumulate them
        print(f"\n[VALIDATE] Aggregating per-category metrics across {len(reward_tensor_dict_lst)} batches...")
        if len(reward_tensor_dict_lst) > 0:
            category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial', 'unknown']
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
                print(f"[VALIDATE] Found {len(category_aggregates)} categories with data")
                for cat_name in sorted(category_aggregates.keys()):
                    agg = category_aggregates[cat_name]
                    if agg['count'] > 0:
                        metric_dict[f'val/{cat_name}_f1'] = agg['f1_sum'] / agg['count']
                        metric_dict[f'val/{cat_name}_bleu'] = agg['bleu_sum'] / agg['count']
                        metric_dict[f'val/{cat_name}_count'] = agg['count']
                        print(f"[VALIDATE] {cat_name}: F1={metric_dict[f'val/{cat_name}_f1']:.4f}, BLEU={metric_dict[f'val/{cat_name}_bleu']:.4f}, count={metric_dict[f'val/{cat_name}_count']:.0f}")
            else:
                print(f"[VALIDATE] Warning: No category data found in any batch")
        else:
            print(f"[VALIDATE] No batches with category data")
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['val/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['val/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['val/num_turns/min'] = num_turns_tensor.min().item()
            print(f"[VALIDATE] num_turns: mean={metric_dict['val/num_turns/mean']:.2f}, max={metric_dict['val/num_turns/max']}, min={metric_dict['val/num_turns/min']}")
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['val/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['val/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['val/completion_tokens/min'] = completion_tokens_tensor.min().item()
            print(f"[VALIDATE] completion_tokens: mean={metric_dict['val/completion_tokens/mean']:.2f}, max={metric_dict['val/completion_tokens/max']}, min={metric_dict['val/completion_tokens/min']}")

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            print(f"\n[VALIDATE] Saving validation generations...")
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'val_step_{self.global_steps}.jsonl'
            
            # Concatenate history lists from different batches
            all_histories = []
            for history_batch in history_lst:
                all_histories.extend(history_batch)
            
            results_to_save = []
            for inp, outp, gt, hist, score in zip(sample_inputs, sample_outputs, sample_groundtruths, all_histories, sample_scores):
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
            print(f"[VALIDATE] Saved {len(results_to_save)} validation results to {output_file}")

        print(f"\n[VALIDATE] Validation complete. Final metrics: {metric_dict}")
        print("="*80 + "\n")
        return metric_dict

    def _test(self):
        """Test pipeline - runs multi-turn generation on all batches and evaluates QA only for finished conversations (non-empty qa_pairs_json)"""
        print("\n" + "="*80)
        print("STARTING TEST")
        print("="*80)
        
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
        print(f"\n[TEST] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
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
            print(f"[TEST] Multi-turn mode enabled")
            print(f"[TEST] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[TEST] agent_roles: {rollout_meta_info['agent_roles']}")
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
            print(f"[TEST] Single-turn mode enabled")
            print(f"[TEST] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[TEST] agent_roles: {rollout_meta_info['agent_roles']}")

        print(f"\n[TEST] Starting test loop: {len(self.test_dataloader)} batches")

        print(f"[TEST] Strategy: Check qa_pairs_json for each sample - if non-empty, conversation has ended and will be evaluated")
        total_batches = len(self.test_dataloader)
        
        for batch_idx, test_data in enumerate(self.test_dataloader):
            print(f"\n{'*'*80}")
            print(f"TEST BATCH {batch_idx + 1}/{total_batches}")
            print(f"{'*'*80}")
            print(f"\n[TEST BATCH {batch_idx + 1}] Creating batch from dataloader...")
            print(f"[TEST BATCH {batch_idx + 1}] Batch size: {len(test_data['question'])}")
            print(f"[TEST BATCH {batch_idx + 1}] test_data keys: {list(test_data.keys())}")
            
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_data['epoch'] = torch.full((len(test_data['question']),), self.global_steps, dtype=torch.long)
            
            # Add test epoch/split info
            # rollout_meta_info['epoch'] = self.global_steps
            rollout_meta_info['split'] = 'test'
            
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)
            print(f"[TEST BATCH {batch_idx + 1}] test_batch.batch keys: {list(test_batch.batch.keys())}")
            print(f"[TEST BATCH {batch_idx + 1}] test_batch.non_tensor_batch keys: {list(test_batch.non_tensor_batch.keys())}")

            # Check which samples have finished (non-zero num_questions) BEFORE repeating
            num_questions_list = test_batch.non_tensor_batch['num_qas']
            finished_mask = [num_questions > 0 for num_questions in num_questions_list]
            num_finished = sum(finished_mask)
            print(f"[TEST BATCH {batch_idx + 1}] Found {num_finished}/{len(finished_mask)} finished conversations (with non-empty qa_pairs_json)")

            # Store original inputs and ground truths BEFORE repeating
            input_texts = test_batch.non_tensor_batch['question']
            sample_inputs.extend(input_texts)
            print(f"[TEST BATCH {batch_idx + 1}] Collected {len(input_texts)} input texts")

            # Store original ground truth if available
            ground_truths = [json.loads(x)[0]["answer"] if json.loads(x) else "N/A" for x in test_batch.non_tensor_batch['qa_pairs_json']]
            sample_groundtruths.extend(ground_truths)
            print(f"[TEST BATCH {batch_idx + 1}] Collected {len(ground_truths)} ground truths")

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)
            print(f"[TEST BATCH {batch_idx + 1}] Repeated batch {self.config.actor_rollout_ref.rollout.val_kwargs.n} times. New size: {len(test_batch.batch)}")
            
            # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
            rollout_idx = torch.arange(0, len(test_batch.batch))
            test_batch.batch['rollout_idx'] = rollout_idx
            
            # we only do test on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                print(f"[TEST BATCH {batch_idx + 1}] Skipping model-based reward model test")
                return {}

            print(f"\n[TEST BATCH {batch_idx + 1}] Preparing generation batch...")
            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('multi_modal_inputs test not implemented yet')
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['rollout_idx', 'batch_idx', 'epoch'], 
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'], 
                        deepcopy=True
                    )
            
            print(f"[TEST BATCH {batch_idx + 1}] Generation batch prepared with {len(test_gen_batch.batch)} samples")
            
            test_gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            print(f'[TEST BATCH {batch_idx + 1}] test_gen_batch meta_info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            print(f"\n[TEST BATCH {batch_idx + 1}] Padding to be divisible by world_size={self.actor_rollout_wg.world_size}...")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            print(f"[TEST BATCH {batch_idx + 1}] Padded batch size: {len(test_gen_batch_padded.batch)}, pad_size: {pad_size}")
            
            print(f"[TEST BATCH {batch_idx + 1}] >>> Calling multi_turn_generate_sequences...")
            test_output_gen_batch_padded = self.actor_rollout_wg.multi_turn_generate_sequences(test_gen_batch_padded)
            print(f"[TEST BATCH {batch_idx + 1}] <<< Generation complete")

            # unpad
            print(f"[TEST BATCH {batch_idx + 1}] Unpadding batch...")
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print(f'[TEST BATCH {batch_idx + 1}] Test generation end. Output batch size: {len(test_output_gen_batch.batch)}')

            # Store generated outputs
            print(f"\n[TEST BATCH {batch_idx + 1}] Processing outputs...")
            output_texts = test_output_gen_batch.non_tensor_batch['response']
            sample_outputs.extend(output_texts)
            print(f"[TEST BATCH {batch_idx + 1}] Collected {len(output_texts)} output texts")
            print(f"[TEST BATCH {batch_idx + 1}] Sample output[0]: {output_texts[0][:100] if isinstance(output_texts[0], str) else output_texts[0]}...")

            history_lst.append(test_output_gen_batch.non_tensor_batch['history'].tolist())
            
            # Collect generation metrics (num_turns, completion_tokens) from ALL batches
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            print(f"[TEST BATCH {batch_idx + 1}] num_turns (all samples): min={num_turns.min()}, max={num_turns.max()}, mean={num_turns.float().mean()}")
            
            turn_level_completion_tokens = test_output_gen_batch.batch['meta_thinking_num_gen_tokens'].cpu() + \
                test_output_gen_batch.batch['reasoning_num_gen_tokens'].cpu()
            completion_tokens = turn_level_completion_tokens.sum(dim=-1)
            completion_tokens_lst.append(completion_tokens)
            print(f"[TEST BATCH {batch_idx + 1}] completion_tokens (all samples): min={completion_tokens.min()}, max={completion_tokens.max()}, mean={completion_tokens.float().mean()}")
            
            # Check if any conversations finished in this batch and evaluate them
            if num_finished > 0:
                print(f"[TEST BATCH {batch_idx + 1}] Evaluating {num_finished} finished conversations...")
                print(f"[TEST BATCH {batch_idx + 1}] Merging generation output with test batch...")
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
                
                print(f"[TEST BATCH {batch_idx + 1}] Filtering to {len(finished_indices)} samples (finished conversations after repeating)")
                
                # Select only finished samples
                finished_test_batch = test_batch_with_gen[finished_indices]
                
                # Compute rewards for finished conversations
                print(f"[TEST BATCH {batch_idx + 1}] Computing rewards for finished conversations...")
                reward_tensor = self.val_reward_fn(finished_test_batch)
                print(f"[TEST BATCH {batch_idx + 1}] Reward tensor keys: {list(reward_tensor.keys())}")
                
                reward_tensor_lst.append(reward_tensor['reasoning_turn_level_reward'])
                reward_tensor_dict_lst.append(reward_tensor)  # Store full dict
                acc_tensor_lst.append(reward_tensor['acc'])
                bleu_tensor_lst.append(reward_tensor['bleu'])
                print(f"[TEST BATCH {batch_idx + 1}] reasoning_turn_level_reward shape: {reward_tensor['reasoning_turn_level_reward'].shape}")
                print(f"[TEST BATCH {batch_idx + 1}] acc shape: {reward_tensor['acc'].shape}")
                print(f"[TEST BATCH {batch_idx + 1}] bleu shape: {reward_tensor['bleu'].shape}")
                
                # Store scores
                scores = reward_tensor['reasoning_turn_level_reward'].sum(-1).cpu().tolist()
                sample_scores.extend(scores)
                print(f"[TEST BATCH {batch_idx + 1}] Sample scores[0]: {scores[0]}")
                
                # Get data sources from finished samples
                data_source_lst.append(finished_test_batch.non_tensor_batch.get('subset', ['locomo'] * reward_tensor['reasoning_turn_level_reward'].shape[0]))
            else:
                print(f"[TEST BATCH {batch_idx + 1}] No finished conversations in this batch, skipping reward computation")
            
            print(f"[TEST BATCH {batch_idx + 1}] Batch processing complete\n")

        # Now compute final metrics from all finished conversations across all batches
        print(f"\n[TEST] All batches processed. Computing final metrics...")
        
        if len(reward_tensor_lst) == 0:
            print("[TEST] WARNING: No finished conversations found across all batches!")
            return {}
        
        # Log generations
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, groundtruths=sample_groundtruths, histories=history_lst)

        # Concatenate all accumulated tensors
        print(f"[TEST] Concatenating reward tensors from {len(reward_tensor_lst)} batches with finished conversations...")
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (total_finished_samples,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        bleu_tensor = torch.cat(bleu_tensor_lst, dim=0).cpu()  # (total_finished_samples,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        print(f"[TEST] Total finished samples evaluated: {reward_tensor.shape[0]}")
        print(f"[TEST] Mean reward: {reward_tensor.mean().item():.4f}")
        print(f"[TEST] Mean accuracy: {acc_tensor.mean().item():.4f}")
        print(f"[TEST] Mean BLEU: {bleu_tensor.mean().item():.4f}")

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
            print(f"[TEST] {data_source} mean reward: {np.mean(rewards):.4f}")
        for data_source, accs in data_source_acc.items():
            metric_dict[f'test/acc/{data_source}'] = np.mean(accs)
            print(f"[TEST] {data_source} mean accuracy: {np.mean(accs):.4f}")
        for data_source, bleus in data_source_bleu.items():
            metric_dict[f'test/bleu/{data_source}'] = np.mean(bleus)
            print(f"[TEST] {data_source} mean BLEU: {np.mean(bleus):.4f}")
        
        # Stage 2 aggregation: Combine per-category metrics across ALL test batches
        # Each batch returns sum and count (not averages), so we just accumulate them
        print(f"\n[TEST] Aggregating per-category metrics across {len(reward_tensor_dict_lst)} batches...")
        if len(reward_tensor_dict_lst) > 0:
            category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial', 'unknown']
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
                print(f"[TEST] Found {len(category_aggregates)} categories with data")
                for cat_name in sorted(category_aggregates.keys()):
                    agg = category_aggregates[cat_name]
                    if agg['count'] > 0:
                        metric_dict[f'test/{cat_name}_f1'] = agg['f1_sum'] / agg['count']
                        metric_dict[f'test/{cat_name}_bleu'] = agg['bleu_sum'] / agg['count']
                        metric_dict[f'test/{cat_name}_count'] = agg['count']
                        print(f"[TEST] {cat_name}: F1={metric_dict[f'test/{cat_name}_f1']:.4f}, BLEU={metric_dict[f'test/{cat_name}_bleu']:.4f}, count={metric_dict[f'test/{cat_name}_count']:.0f}")
            else:
                print(f"[TEST] Warning: No category data found in any batch")
        else:
            print(f"[TEST] No batches with category data")
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['test/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['test/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['test/num_turns/min'] = num_turns_tensor.min().item()
            print(f"[TEST] num_turns: mean={metric_dict['test/num_turns/mean']:.2f}, max={metric_dict['test/num_turns/max']}, min={metric_dict['test/num_turns/min']}")
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['test/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['test/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['test/completion_tokens/min'] = completion_tokens_tensor.min().item()
            print(f"[TEST] completion_tokens: mean={metric_dict['test/completion_tokens/mean']:.2f}, max={metric_dict['test/completion_tokens/max']}, min={metric_dict['test/completion_tokens/min']}")

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            print(f"\n[TEST] Saving test generations...")
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'test_step_{self.global_steps}.jsonl'
            
            # Concatenate history lists from different batches
            all_histories = []
            for history_batch in history_lst:
                all_histories.extend(history_batch)
            
            results_to_save = []
            for inp, outp, gt, hist, score in zip(sample_inputs, sample_outputs, sample_groundtruths, all_histories, sample_scores):
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
            print(f"[TEST] Saved {len(results_to_save)} test results to {output_file}")

        print(f"\n[TEST] Test complete. Final metrics: {metric_dict}")
        print("="*80 + "\n")
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

        print("\n" + "="*80)
        print("STARTING PPO TRAINING LOOP (fit method)")
        print("="*80)

        self.global_steps = 0

        # load checkpoint before doing anything
        print("\n[FIT] Loading checkpoint...")
        self._load_checkpoint()
        print(f"[FIT] Checkpoint loaded. Starting from global_steps={self.global_steps}")

        if self.config.trainer.get('fork_wandb_id', None) is not None:
            fork_wandb_id = self.config.trainer.fork_wandb_id
            # wandb_kwargs = {'resume': 'must', 'id': fork_wandb_id}
            print(f'**[WANDB]: will fork run from wandb id: `{fork_wandb_id}` at step {self.global_steps} **')
            
            # e.g. fork_from="6yaq69uw?_step=200"
            wandb_kwargs = {'fork_from': f"{fork_wandb_id}?_step={self.global_steps}"}
        else:
            wandb_kwargs = {}
        
        print(f"\n[FIT] Initializing logger: {self.config.trainer.logger}")
        print(f"[FIT] Project: {self.config.trainer.project_name}, Experiment: {self.config.trainer.experiment_name}")
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_kwargs=wandb_kwargs
                          )

        # perform test if test_only mode is enabled
        if self.val_reward_fn is not None and self.config.trainer.get('test_only', False):
            print("\n[FIT] Test-only mode enabled. Running test and exiting...")
            test_metrics = self._test()
            pprint(f'Test metrics: {test_metrics}')
            logger.log(data=test_metrics, step=self.global_steps)
            print("[FIT] Test-only mode complete. Exiting.")
            return

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            print("\n[FIT] Running initial validation before training...")
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                print("[FIT] Val-only mode enabled. Exiting after validation.")
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        print(f"\n[FIT] Configuring rollout meta_info with max_num_turns={max_num_turns}")
        if max_num_turns > 1:
            from prompt.math.multi_turn_mamrp import MEMORY_REASONER_PROMPT, MEMORY_EXECUTOR_PROMPT
            from prompt import FINISH_FLAG
            rollout_meta_info = {
                'agent_roles': ['meta_thinking', 'reasoning'],
                'finish_flag': None, # changed this to None from FINISH_FLAG
                'system_prompts': {
                    'meta_thinking': MEMORY_REASONER_PROMPT,
                    'reasoning': MEMORY_EXECUTOR_PROMPT
                },
                'max_num_turns': max_num_turns
            }
            print(f"[FIT] Multi-turn mode enabled with FINISH_FLAG")
            print(f"[FIT] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[FIT] agent_roles: {rollout_meta_info['agent_roles']}")
            print(f"[FIT] finish_flag: {rollout_meta_info['finish_flag'][:50] if rollout_meta_info['finish_flag'] else None}...")
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
            print(f"[FIT] Single-turn mode enabled (no FINISH_FLAG)")
            print(f"[FIT] rollout_meta_info keys: {list(rollout_meta_info.keys())}")
            print(f"[FIT] agent_roles: {rollout_meta_info['agent_roles']}")
        
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_prompt_cnt = 0 
        all_negative_cnt = 0
        all_positive_cnt = 0

        print(f"\n[FIT] Starting training loop: {self.config.trainer.total_epochs} epochs, {len(self.train_dataloader)} batches per epoch")
        print(f"[FIT] Total training steps: {self.total_training_steps}")

        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{self.config.trainer.total_epochs}")
            print(f"{'='*80}")
            for batch_dict in self.train_dataloader:
                print(f"\n{'*'*80}")
                print(f"TRAINING STEP {self.global_steps}/{self.total_training_steps}")
                print(f"{'*'*80}")
                metrics = {}
                timing_raw = {}

                print("batch_dict[chunk_id]: ", batch_dict['chunk_id'] if 'chunk_id' in batch_dict else 'N/A')

                # create a dummy tensor for the construction function
                print(f"\n[STEP {self.global_steps}] Creating batch from dataloader...")
                print(f"[STEP {self.global_steps}] Batch size: {len(batch_dict['question'])}")
                print(f"[STEP {self.global_steps}] batch_dict keys: {list(batch_dict.keys())}")
                print(f"[STEP {self.global_steps}] Sample question[0]: {batch_dict['question'][0][:100] if isinstance(batch_dict['question'][0], str) else batch_dict['question'][0]}...")
                if 'reward_model' in batch_dict:
                    print(f"[STEP {self.global_steps}] reward_model[0] keys: {list(batch_dict['reward_model'][0].keys()) if isinstance(batch_dict['reward_model'][0], dict) else 'N/A'}")
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
                        print(f"[STEP {self.global_steps}] Adding {len(idxs)} indices to replay buffer (step={self.global_steps})")
                        print(f"[STEP {self.global_steps}] idxs: {idxs}")
                        self.replay_buffer.add_indices(idxs, epoch=epoch)
                    except Exception:
                        pass

                # sample from replay and merge with current batch if requested
                replay_ratio = float(self.config.trainer.get('replay_mix_ratio', 0.5))
                n_replay = int(len(batch_dict['question']) * replay_ratio)
                if n_replay > 0 and getattr(self, 'replay_buffer', None) and len(self.replay_buffer.buffer) > 0:
                    sampled = self.replay_buffer.sample(n_replay, strategy=self.config.trainer.get('replay_strategy', 'uniform'))
                    print(f"[STEP {self.global_steps}] Sampling {len(sampled)} entries from replay buffer to merge into current batch")
                    if len(sampled) > 0:
                        # sampled contains tuples (index, orig_epoch)
                        sample_indices = [s[0] for s in sampled]
                        sample_epochs = [s[1] for s in sampled]
                        print(f"[STEP {self.global_steps}] sample_indices: {sample_indices}, sample_epochs: {sample_epochs}")
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
                print(f"[STEP {self.global_steps}] new_batch.batch keys: {list(new_batch.batch.keys())}")
                print(f"[STEP {self.global_steps}] new_batch.non_tensor_batch keys: {list(new_batch.non_tensor_batch.keys())}")
                print(f"[STEP {self.global_steps}] new_batch.meta_info keys: {list(new_batch.meta_info.keys())}")
                if 'batch_idx' in new_batch.batch:
                    print(f"[STEP {self.global_steps}] batch_idx shape: {new_batch.batch['batch_idx'].shape}")
                new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # save rollout idx to use it in memory management (AFTER repeating to get unique indices for each rollout)
                rollout_idx = torch.arange(0, len(new_batch.batch))
                new_batch.batch['rollout_idx'] = rollout_idx

                num_gen_batches += 1

                # pop those keys for generation
                print(f"\n[STEP {self.global_steps}] Preparing generation batch...")
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
                        non_tensor_batch_keys=['sample_id', 'chunk_id', 'speakers', 'qa_pairs_json', 'num_qas', 'turns_json', 'session_id', 'session_time', 'session_evidences_json'], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts', 'max_num_turns', 'split'],
                        deepcopy=True
                    )
                print(f"[STEP {self.global_steps}] Generation batch prepared with {len(gen_batch.batch)} samples")
                print(f"[STEP {self.global_steps}] gen_batch.batch keys: {list(gen_batch.batch.keys())}")
                print(f"[STEP {self.global_steps}] gen_batch.non_tensor_batch keys: {list(gen_batch.non_tensor_batch.keys())}")
                print(f"[STEP {self.global_steps}] gen_batch.meta_info keys: {list(gen_batch.meta_info.keys())}")

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    print(f"\n[STEP {self.global_steps}] >>> Calling multi_turn_generate_sequences...")
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.multi_turn_generate_sequences(gen_batch)
                    print(f"[STEP {self.global_steps}] <<< Generation complete. Output batch size: {len(gen_batch_output.batch)}")
                    print(f"[STEP {self.global_steps}] gen_batch_output.batch keys: {list(gen_batch_output.batch.keys())}")
                    print(f"[STEP {self.global_steps}] gen_batch_output.non_tensor_batch keys: {list(gen_batch_output.non_tensor_batch.keys())}")
                    if 'meta_thinking_attention_mask' in gen_batch_output.batch:
                        print(f"[STEP {self.global_steps}] meta_thinking_attention_mask shape: {gen_batch_output.batch['meta_thinking_attention_mask'].shape}")
                    if 'reasoning_attention_mask' in gen_batch_output.batch:
                        print(f"[STEP {self.global_steps}] reasoning_attention_mask shape: {gen_batch_output.batch['reasoning_attention_mask'].shape}")
                    if 'response' in gen_batch_output.non_tensor_batch:
                        print(f"[STEP {self.global_steps}] Sample response[0]: {gen_batch_output.non_tensor_batch['response'][0][:100] if isinstance(gen_batch_output.non_tensor_batch['response'][0], str) else gen_batch_output.non_tensor_batch['response'][0]}...")
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError('REMAX is not implemented yet')
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    print(f"\n[STEP {self.global_steps}] Merging generation output with original batch...")
                    print(f"[STEP {self.global_steps}] BEFORE union - gen_batch_output.batch keys: {list(gen_batch_output.batch.keys())}")                    
                    new_batch = new_batch.union(gen_batch_output)
                    print(f"[STEP {self.global_steps}] AFTER union - Merged batch size: {len(new_batch.batch)}")
                    print(f"[STEP {self.global_steps}] AFTER union - Merged new_batch.batch keys: {list(new_batch.batch.keys())}")
                    print(f"[STEP {self.global_steps}] Merged new_batch.non_tensor_batch keys: {list(new_batch.non_tensor_batch.keys())}")
                    if 'num_turns' in new_batch.non_tensor_batch:
                        print(f"[STEP {self.global_steps}] Sample num_turns[0]: {new_batch.non_tensor_batch['num_turns'][0]}")

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        print(f"[STEP {self.global_steps}] Balancing batch across DP ranks...")
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


                    # compute values
                    if self.use_critic:
                        raise NotImplementedError('critic is not implemented yet')
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('reward', timing_raw):
                        print(f"\n[STEP {self.global_steps}] Computing rewards...")
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            raise NotImplementedError('RM is not implemented yet')
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # add mask_unfinished_reward to meta_info
                        new_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                        new_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                        print(f"[STEP {self.global_steps}] Calling reward_fn (rule-based)...")
                        # rule-based rm build token-level reward_tensor_map for each agent
                        # {
                        #     "meta_thinking_turn_level_reward": tensor([...], device='cuda:0'),
                        #     "reasoning_turn_level_reward": tensor([...], device='cuda:0'),
                        # }
                        reward_tensor_map = self.reward_fn(new_batch)
                        print(f"[STEP {self.global_steps}] Reward computed. Keys: {list(reward_tensor_map.keys())}")
                        
                        # Extract per-category metrics for training (report per-batch, not accumulated)
                        category_names = ['multi_hop', 'single_hop', 'temporal', 'open_domain', 'adversarial', 'unknown']
                        for cat_name in category_names:
                            f1_sum_key = f'{cat_name}_f1_sum'
                            if f1_sum_key in reward_tensor_map:
                                # Compute average from sum and count for this batch
                                f1_sum = reward_tensor_map.pop(f1_sum_key).item()
                                bleu_sum = reward_tensor_map.pop(f'{cat_name}_bleu_sum').item()
                                count = int(reward_tensor_map.pop(f'{cat_name}_count').item())
                                
                                if count > 0:
                                    # Report batch-level average (not accumulated)
                                    metrics[f'train/{cat_name}_f1'] = f1_sum / count
                                    metrics[f'train/{cat_name}_bleu'] = bleu_sum / count
                                    metrics[f'train/{cat_name}_count'] = count
                                    print(f"[STEP {self.global_steps}] Batch category {cat_name}: F1={metrics[f'train/{cat_name}_f1']:.4f}, BLEU={metrics[f'train/{cat_name}_bleu']:.4f}, count={count}")
                        
                        # for key in reward_tensor_map.keys():
                        #     if isinstance(reward_tensor_map[key], torch.Tensor):
                        #         print(f"[STEP {self.global_steps}] {key} shape: {reward_tensor_map[key].shape}, dtype: {reward_tensor_map[key].dtype}")
                        #         print(f"[STEP {self.global_steps}] {key} : {reward_tensor_map[key]}")
                        new_batch.batch['acc'] = reward_tensor_map.pop('acc')
                        new_batch.batch['bleu'] = reward_tensor_map.pop('bleu')
                        new_batch.batch['evidence'] = reward_tensor_map.pop('evidence')
                        print(f"[STEP {self.global_steps}] acc shape: {new_batch.batch['acc'].shape}, acc: {new_batch.batch['acc']}")
                        print(f"[STEP {self.global_steps}] bleu shape: {new_batch.batch['bleu'].shape}, bleu: {new_batch.batch['bleu']}")
                        print(f"[STEP {self.global_steps}] evidence shape: {new_batch.batch['evidence'].shape}, evidence: {new_batch.batch['evidence']}")
                        # batch.batch['token_level_scores'] = reward_tensor
                        for key_reward, reward_tensor in reward_tensor_map.items():
                            new_batch.batch[key_reward] = reward_tensor
                            # get_turn_mask, shape (bsz, max_num_turns), 1 for valid turn, 0 for invalid turn
                            turn_mask = verl_F.get_turn_mask(reward_tensor, new_batch.non_tensor_batch['num_turns'])
                            key_return = key_reward.replace('reward', 'return')
                            # compute turn_level return with turn_level_gamma
                            new_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                reward_tensor, turn_mask, self.config.algorithm.gamma_turn_level)
                            # print(f"[STEP {self.global_steps}] After turn level reward/return computation: {key_return}: {new_batch.batch[key_return]}")
                    
                    
                    # statistics for group filter
                    if self.config.actor_rollout_ref.rollout.n > 1:
                        print(f"\n[STEP {self.global_steps}] Computing group filter statistics...")
                        # key_reward = list(reward_tensor_map.keys())[0]
                        # one_agent_reward_tensor = reward_tensor_map[key_reward]
                        acc_tensor = new_batch.batch['acc']
                        id2acc = defaultdict(list)
                        for i_bsz, uid in enumerate(new_batch.non_tensor_batch['uid']):
                            id2acc[uid].append(acc_tensor[i_bsz])
                        print(f"[STEP {self.global_steps}] Grouped {len(id2acc)} unique prompts")

                        kept_prompt_uids = []
                        for key_uid, acc_this_uid in id2acc.items():
                            acc_this_uid = torch.tensor(acc_this_uid)
                            if (acc_this_uid == 0).all():
                                all_negative_cnt += 1
                            elif (acc_this_uid == 1).all():
                                all_positive_cnt += 1
                            else:
                                # keep prompt with none-zero advantages
                                kept_prompt_uids.append(key_uid)
                            total_prompt_cnt += 1
                    
                    if not self.config.algorithm.filter_groups.enable:
                        # if not enable group filter, keep all data
                        print(f"[STEP {self.global_steps}] Group filter disabled. Keeping all data.")
                        batch = new_batch
                    else:
                        print(f"[STEP {self.global_steps}] Group filter enabled. Filtering data...")
                        # filter data based on group filter statistics
                        num_prompt_in_batch += len(kept_prompt_uids)
                        print(f"[STEP {self.global_steps}] Kept {len(kept_prompt_uids)} prompts. Total in batch: {num_prompt_in_batch}")
                        # get kept data batch
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        new_batch = new_batch[kept_traj_idxs]
                        print(f"[STEP {self.global_steps}] Filtered batch to {len(new_batch.batch)} trajectories")
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])
                            print(f"[STEP {self.global_steps}] Concatenated batches. Total size: {len(batch.batch)}")
                        
                        # check if we have enough data
                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            # keep generating
                            print(f'[STEP {self.global_steps}] {num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'[STEP {self.global_steps}] {num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            print(f"[STEP {self.global_steps}] Sufficient data collected. Aligning batch...")
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                            print(f"[STEP {self.global_steps}] Batch aligned to {traj_bsz} trajectories")

                    if self.config.actor_rollout_ref.rollout.n > 1:
                        metrics.update({
                            'rollout/all_negative_cnt': all_negative_cnt,
                            'rollout/all_positive_cnt': all_positive_cnt,
                            'rollout/total_prompt_cnt': total_prompt_cnt,
                            'rollout/num_gen_batches': num_gen_batches,
                            'rollout/acc_mean': new_batch.batch['acc'].mean().item()
                        })
                    
                    with _timer('save_train_generation', timing_raw):
                        # save train generation
                        if self.config.trainer.get('save_train_generations', False):
                            self._save_train_generations(batch)

                    with _timer('adv', timing_raw):
                        print(f"\n[STEP {self.global_steps}] Computing advantages...")
                        # Merge different role data into a single DataProto
                        print(f"[STEP {self.global_steps}] Merging roles data...")
                        print(f"[STEP {self.global_steps}] Before merge - batch.batch keys: {list(batch.batch.keys())}")
                        print(f"[STEP {self.global_steps}] Before merge - batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                        merged_batch = merge_roles_data(batch)
                        print(f"[STEP {self.global_steps}] Merged batch size: {len(merged_batch.batch)}")
                        print(f"[STEP {self.global_steps}] After merge - merged_batch.batch keys: {list(merged_batch.batch.keys())}")
                        print(f"[STEP {self.global_steps}] After merge - merged_batch.non_tensor_batch keys: {list(merged_batch.non_tensor_batch.keys())}")
                        # assign turn_level scores to the last token of each turn, w/ step_ids
                        #  and then i'll call compute_advantage to distribute the score to all
                        #  tokens of each step.
                        print(f"[STEP {self.global_steps}] Computing token-level scores...")
                        token_level_scores = compute_token_level_scores(merged_batch)
                        print(f"[STEP {self.global_steps}] token_level_scores shape: {token_level_scores.shape}, dtype: {token_level_scores.dtype}")
                        print(f"[STEP {self.global_steps}] token_level_scores sample[0, :10]: {token_level_scores[0, :10]}")
                        merged_batch.batch['token_level_scores'] = token_level_scores
                        batch = merged_batch
                        
                        # # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                          kl_ctrl=self.kl_ctrl,
                        #                                          kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        
                        # XXX(ziyu): debug
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # in this case, its usage is changed.
                        # for REINFORCE++, it's used to distribute the score from last token of each turn
                        # to all tokens of each step.
                        # for GRPO, we use turn_level_reward.sum(-1) as the outcome reward and then
                        # assign each label token the normalized advantage.
                        print(f"[STEP {self.global_steps}] Computing advantages with {self.config.algorithm.adv_estimator}...")
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma_token_level,
                                                  lam=self.config.algorithm.lam_token_level,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        print(f"[STEP {self.global_steps}] Advantages computed.")
                        print(f"[STEP {self.global_steps}] Final batch.batch keys: {list(batch.batch.keys())}")
                        if 'advantages' in batch.batch:
                            print(f"[STEP {self.global_steps}] advantages shape: {batch.batch['advantages'].shape}")
                            print(f"[STEP {self.global_steps}] advantages sample[0, :10]: {batch.batch['advantages'][0, :10]}")

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        print(f"\n[STEP {self.global_steps}] Computing old log probabilities...")
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                        print(f"[STEP {self.global_steps}] Old log probs computed.")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            print(f"[STEP {self.global_steps}] Computing reference log probabilities...")
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            print(f"[STEP {self.global_steps}] Reference log probs computed.")


                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            print(f"\n[STEP {self.global_steps}] Updating critic...")
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        print(f"[STEP {self.global_steps}] Critic updated.")

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            print(f"\n[STEP {self.global_steps}] Updating actor...")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        print(f"[STEP {self.global_steps}] Actor updated.")
                    else:
                        print(f"\n[STEP {self.global_steps}] Skipping actor update (warmup: {self.global_steps}/{self.config.trainer.critic_warmup})")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            print(f"\n[STEP {self.global_steps}] Running validation...")
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            print(f"[STEP {self.global_steps}] Validation complete.")
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            print(f"\n[STEP {self.global_steps}] Saving checkpoint...")
                            self._save_checkpoint()
                            print(f"[STEP {self.global_steps}] Checkpoint saved.")

                # collect metrics
                print(f"\n[STEP {self.global_steps}] Computing and logging metrics...")
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                print(f"[STEP {self.global_steps}] Metrics logged. Step complete.\n")

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                all_negative_cnt = 0
                all_positive_cnt = 0
                total_prompt_cnt = 0

                if is_last_step:
                    print(f"\n{'='*80}")
                    print(f"TRAINING COMPLETE")
                    print(f"{'='*80}")
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
                print(f"[FIT] Incrementing global_steps to {self.global_steps}")

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