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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import Iterator


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class ChunkBatchSampler(Sampler):
    """
    Custom batch sampler that ensures batches never span multiple chunk_ids.
    
    For training: Pads incomplete batches by repeating samples from the same chunk (acts like extra rollouts)
    For validation/test: Returns actual batch sizes without padding (pad_incomplete=False)
    """
    def __init__(self, dataset: 'RLHFDataset', batch_size: int, drop_last: bool = False, pad_incomplete: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pad_incomplete = pad_incomplete  # Whether to pad batches with repetitions
        
        if not hasattr(dataset, 'chunk_groups') or dataset.chunk_groups is None:
            raise ValueError("Dataset must have chunk_groups (requires chunk_id column)")
        
        self.chunk_groups = dataset.chunk_groups
        self.chunk_ids = sorted(self.chunk_groups.keys())
        
    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches that respect chunk boundaries"""
        for chunk_id in self.chunk_ids:
            chunk_indices = list(self.chunk_groups[chunk_id])
            chunk_size = len(chunk_indices)
            
            # Process this chunk in batches
            i = 0
            while i < chunk_size:
                batch_indices = []
                
                # Collect samples for this batch
                for j in range(self.batch_size):
                    if i + j < chunk_size:
                        batch_indices.append(chunk_indices[i + j])
                    elif self.pad_incomplete:
                        # For training: pad with repeated samples from same chunk (acts like extra rollouts)
                        repeat_idx = (i + j) % chunk_size
                        batch_indices.append(chunk_indices[repeat_idx])
                    # For validation/test: don't pad, just use what we have
                
                # Yield batch if we should keep it
                if len(batch_indices) == self.batch_size:
                    # Full batch - always yield
                    yield batch_indices
                elif not self.drop_last:
                    # Incomplete batch - yield if drop_last=False
                    yield batch_indices
                # else: drop incomplete batch (drop_last=True)
                
                i += self.batch_size
    
    def __len__(self) -> int:
        """Calculate total number of batches"""
        total_batches = 0
        for chunk_id in self.chunk_ids:
            chunk_size = len(self.chunk_groups[chunk_id])
            if self.drop_last:
                total_batches += chunk_size // self.batch_size
            else:
                total_batches += (chunk_size + self.batch_size - 1) // self.batch_size
        return total_batches


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 prompt_key='question',
                 shuffle=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.prompt_key = prompt_key
        self.shuffle = shuffle

        # Whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # Read parquet files
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        
        # Sort by chunk_id first, then sample_id to ensure batches contain same chunk_id
        # Only sort by chunk_id when shuffle is False to avoid cross-chunk memory issues
        if not self.shuffle and 'chunk_id' in self.dataframe.columns and 'sample_id' in self.dataframe.columns:
            # Ensure chunk_id is numeric for proper sorting (not string sorting)
            self.dataframe['chunk_id'] = pd.to_numeric(self.dataframe['chunk_id'], errors='coerce')
            self.dataframe = self.dataframe.sort_values(by=['chunk_id', 'sample_id']).reset_index(drop=True)
            
            # Group by chunk_id and get counts
            chunk_counts = self.dataframe.groupby('chunk_id').size()
            print(f'dataset len: {len(self.dataframe)} (sorted by chunk_id, sample_id)')
            print(f'Chunk ID counts: {dict(chunk_counts)}')
            
            # Store chunk_id groups for repeat logic
            self.chunk_groups = self.dataframe.groupby('chunk_id').indices
        elif 'chunk_id' in self.dataframe.columns:
            # Still need chunk_groups even without shuffle for ChunkBatchSampler
            self.dataframe['chunk_id'] = pd.to_numeric(self.dataframe['chunk_id'], errors='coerce')
            self.chunk_groups = self.dataframe.groupby('chunk_id').indices
            print(f'dataset len: {len(self.dataframe)} (chunk_groups created, no sorting)')
        else:
            self.chunk_groups = None
            print(f'dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Return the prompt data directly without tokenization
        If accessing beyond the length of a chunk_id group, cycle within that group
        """
        # If chunk_groups exist and we have chunk_id info, handle repeating
        if self.chunk_groups is not None and item < len(self.dataframe):
            row = self.dataframe.iloc[item]
            chunk_id = row['chunk_id']
            chunk_indices = self.chunk_groups[chunk_id]
            
            # Find position within this chunk's group
            relative_pos = list(chunk_indices).index(item)
            # Use modulo to cycle within the same chunk if needed (for future batching logic)
            actual_index = chunk_indices[relative_pos % len(chunk_indices)]
            
            row_dict: dict = self.dataframe.iloc[actual_index].to_dict()
        else:
            row_dict: dict = self.dataframe.iloc[item].to_dict()
        
        # Get and pop the prompt from the row dictionary
        question = row_dict.pop(self.prompt_key)
        
        # Store the raw chat data
        row_dict['question'] = question
        
        # Add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = item

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    
    
if __name__ == "__main__":
    dataset = RLHFDataset(parquet_files=["data/MATH/train.parquet"], prompt_key="question")
    print(len(dataset))
    print(dataset[0])
