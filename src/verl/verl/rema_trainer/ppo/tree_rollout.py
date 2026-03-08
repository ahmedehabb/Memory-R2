"""
Tree-based rollout management for multi-turn GRPO with pruning.

Instead of flat n-independent rollouts, this module manages a tree structure where
rollouts branch at session boundaries. Children of the same parent share the same
initial state (memory snapshot), forming valid GRPO comparison groups.

A pruning budget keeps the total number of active leaves bounded.

References:
- MURPHY: Multi-Turn Reinforcement Learning with Pruning and Hierarchical Yields
- GRPO: Group Relative Policy Optimization
"""

from __future__ import annotations

import copy
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class TreeNode:
    """A single node in the rollout tree.

    Each node represents one rollout at one session. Children are rollouts
    branched from this node's memory state at the next session.
    """

    node_id: int  # Unique identifier across the whole tree
    parent_id: Optional[int]  # None for root nodes
    sample_id: str  # Conversation / prompt ID
    session_idx: int  # Which session (0-based)
    rollout_idx: int  # The rollout_idx used in the batch for this node

    # Populated after generation / reward
    reward: float = 0.0
    intermediate_signal: float = 0.0  # evidence_coverage + compression for pruning

    # Tree links (populated by RolloutTree)
    children: List[int] = field(default_factory=list)  # node_ids of children
    is_alive: bool = True  # False if pruned


class RolloutTree:
    """Manages a tree of rollouts for one or more prompts.

    Usage flow per training step:
        1. ``create_roots()`` – initialise root nodes at session 0
        2. For each subsequent session:
           a. ``branch()``       – create children from surviving leaves
           b. ``prune_to_budget()`` – trim to budget
        3. After terminal rewards:
           a. ``get_grpo_groups()`` – group siblings for advantage computation
    """

    def __init__(
        self,
        initial_rollouts: int = 8,
        branch_factor: int = 2,
        max_active_leaves: int = 16,
        pruning_strategy: str = "variance",
        alpha1: float = 0.0,
        alpha2: float = 1.0,
    ):
        self.initial_rollouts = initial_rollouts
        self.branch_factor = branch_factor
        self.max_active_leaves = max_active_leaves
        self.pruning_strategy = pruning_strategy
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.nodes: Dict[int, TreeNode] = {}
        self._next_node_id = 0
        # Mapping: sample_id -> list of node_ids at the current leaf level
        self._active_leaves: Dict[str, List[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _alloc_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def create_roots(self, sample_ids: List[str]) -> List[TreeNode]:
        """Create root-level nodes (session 0) for each prompt.

        Args:
            sample_ids: Unique prompt identifiers.

        Returns:
            Flat list of all created root nodes (len = len(sample_ids) * initial_rollouts).
        """
        roots = []
        for sid in sample_ids:
            for r in range(self.initial_rollouts):
                node = TreeNode(
                    node_id=self._alloc_id(),
                    parent_id=None,
                    sample_id=sid,
                    session_idx=0,
                    rollout_idx=r,
                )
                self.nodes[node.node_id] = node
                self._active_leaves[sid].append(node.node_id)
                roots.append(node)
        return roots

    def branch(self, session_idx: int) -> List[TreeNode]:
        """Branch each surviving leaf into ``branch_factor`` children.

        Args:
            session_idx: The new session index for the children.

        Returns:
            List of all newly created child nodes.
        """
        new_leaves: Dict[str, List[int]] = defaultdict(list)
        all_new_nodes = []
        rollout_counter: Dict[str, int] = defaultdict(int)

        for sid, leaf_ids in self._active_leaves.items():
            for leaf_id in leaf_ids:
                parent = self.nodes[leaf_id]
                if not parent.is_alive:
                    continue
                for _ in range(self.branch_factor):
                    child = TreeNode(
                        node_id=self._alloc_id(),
                        parent_id=leaf_id,
                        sample_id=sid,
                        session_idx=session_idx,
                        rollout_idx=rollout_counter[sid],
                    )
                    rollout_counter[sid] += 1
                    self.nodes[child.node_id] = child
                    parent.children.append(child.node_id)
                    new_leaves[sid].append(child.node_id)
                    all_new_nodes.append(child)

        self._active_leaves = new_leaves
        return all_new_nodes

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def set_intermediate_signals(self, node_id_to_signal: Dict[int, float]):
        """Set intermediate pruning signals for nodes (e.g. evidence coverage)."""
        for nid, sig in node_id_to_signal.items():
            if nid in self.nodes:
                self.nodes[nid].intermediate_signal = sig

    def set_rewards(self, node_id_to_reward: Dict[int, float]):
        """Set rewards for nodes (used for final GRPO advantage)."""
        for nid, r in node_id_to_reward.items():
            if nid in self.nodes:
                self.nodes[nid].reward = r

    def prune_to_budget(self, budget: Optional[int] = None) -> List[int]:
        """Prune active leaves to at most ``budget`` per sample_id.

        Uses the configured pruning strategy to decide which nodes to keep.

        Args:
            budget: Max leaves to keep *per sample_id*. Defaults to
                    ``max_active_leaves`` divided by the number of unique sample_ids.

        Returns:
            List of node_ids that were pruned (killed).
        """
        n_samples = max(len(self._active_leaves), 1)
        if budget is None:
            budget = max(1, self.max_active_leaves // n_samples)

        pruned_ids = []

        for sid, leaf_ids in self._active_leaves.items():
            alive_ids = [nid for nid in leaf_ids if self.nodes[nid].is_alive]
            if len(alive_ids) <= budget:
                continue

            # --- Choose which to keep ---
            if self.pruning_strategy == "variance":
                keep = self._prune_variance(alive_ids, budget)
            elif self.pruning_strategy == "quality" or self.pruning_strategy == "anchor":
                keep = self._prune_quality(alive_ids, budget)
            elif self.pruning_strategy == "anchor_intra":
                keep = self._prune_intra(alive_ids, budget)
            elif self.pruning_strategy == "ucb":
                keep = self._prune_ucb(alive_ids, budget)
            elif self.pruning_strategy == "intra":
                keep = self._prune_intra(alive_ids, budget)
            elif self.pruning_strategy == "inter":
                keep = self._prune_inter(alive_ids, budget)
            else:
                raise ValueError(f"Unknown pruning strategy: {self.pruning_strategy}")

            keep_set = set(keep)
            for nid in alive_ids:
                if nid not in keep_set:
                    self._kill_subtree(nid)
                    pruned_ids.append(nid)

            # Update active leaves to only kept ones
            self._active_leaves[sid] = [nid for nid in leaf_ids if self.nodes[nid].is_alive]

        return pruned_ids

    def _prune_variance(self, node_ids: List[int], budget: int) -> List[int]:
        """IntraP-style: keep nodes from groups with highest reward variance.

        Groups are defined by parent_id. We rank groups by variance of
        intermediate_signal among their children, keep the top groups, and
        within each kept group keep the children that contribute most to
        the variance (furthest from the group mean).
        """
        # Group by parent
        parent_groups: Dict[Optional[int], List[int]] = defaultdict(list)
        for nid in node_ids:
            parent_groups[self.nodes[nid].parent_id].append(nid)

        # Score each group by variance of intermediate signal
        group_scores = []
        for pid, children in parent_groups.items():
            signals = [self.nodes[c].intermediate_signal for c in children]
            var = float(np.var(signals)) if len(signals) > 1 else 0.0
            group_scores.append((pid, var, children))

        # Sort groups by variance descending
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # Greedily fill budget from highest-variance groups
        keep = []
        for pid, var, children in group_scores:
            if len(keep) >= budget:
                break
            remaining = budget - len(keep)
            if len(children) <= remaining:
                keep.extend(children)
            else:
                # Within this group, keep nodes furthest from the group mean
                signals = np.array([self.nodes[c].intermediate_signal for c in children])
                mean_sig = signals.mean()
                deviations = np.abs(signals - mean_sig)
                top_k_idx = np.argsort(deviations)[::-1][:remaining]
                keep.extend([children[i] for i in top_k_idx])

        return keep[:budget]

    def _prune_intra(self, node_ids: List[int], budget: int) -> List[int]:
        """Intra-Group Pruning: apply limits inside each parent's group individually.
        Ensures groups of at least size 2 are selected to balance GRPO sibling comparisons.
        """
        parent_groups = defaultdict(list)
        for nid in node_ids:
            parent_groups[self.nodes[nid].parent_id].append(nid)

        parent_scores = []
        for pid, children in parent_groups.items():
            score = np.max([self.nodes[c].intermediate_signal for c in children])
            parent_scores.append((pid, float(score), children))
        
        parent_scores.sort(key=lambda x: x[1], reverse=True)

        # target at least 2 children per parent for GRPO logic to work effectively
        children_per_parent = 2 if budget >= 2 else 1
        
        # If budget allows more children per parent while keeping all parents, do so
        num_parents = len(parent_groups)
        if num_parents > 0 and budget // num_parents >= 2:
            children_per_parent = budget // num_parents

        parents_to_keep = budget // children_per_parent

        keep = []
        for pid, score, children in parent_scores[:parents_to_keep]:
            children_sorted = sorted(children, key=lambda c: self.nodes[c].intermediate_signal, reverse=True)
            keep.extend(children_sorted[:children_per_parent])

        # Fill any remainder budget
        remaining = budget - len(keep)
        if remaining > 0 and len(parent_scores) > 0:
            pid, score, children = parent_scores[0]
            children_sorted = sorted(children, key=lambda c: self.nodes[c].intermediate_signal, reverse=True)
            for child in children_sorted:
                if remaining == 0:
                    break
                if child not in keep:
                    keep.append(child)
                    remaining -= 1

        return keep[:budget]

    def _prune_inter(self, node_ids: List[int], budget: int) -> List[int]:
        """Inter-Group Pruning: prune parents entirely, keeping all children of the winners."""
        parent_groups = defaultdict(list)
        for nid in node_ids:
            parent_groups[self.nodes[nid].parent_id].append(nid)

        parent_scores = []
        for pid, children in parent_groups.items():
            score = np.mean([self.nodes[c].intermediate_signal for c in children])
            parent_scores.append((pid, float(score), children))
        
        parent_scores.sort(key=lambda x: x[1], reverse=True)

        keep = []
        for pid, score, children in parent_scores:
            if len(keep) >= budget:
                break
            remaining = budget - len(keep)
            if len(children) <= remaining:
                keep.extend(children)
            else:
                # If a parent's children exceed remaining budget, keep its top children
                children_sorted = sorted(children, key=lambda c: self.nodes[c].intermediate_signal, reverse=True)
                keep.extend(children_sorted[:remaining])

        return keep[:budget]

    def _prune_quality(self, node_ids: List[int], budget: int) -> List[int]:
        """Keep nodes with highest intermediate signal (greedy exploitation)."""
        scored = [(nid, self.nodes[nid].intermediate_signal) for nid in node_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:budget]]

    def _prune_ucb(self, node_ids: List[int], budget: int) -> List[int]:
        """InterP-style: UCB scoring per parent group, keep top groups."""
        parent_groups: Dict[Optional[int], List[int]] = defaultdict(list)
        for nid in node_ids:
            parent_groups[self.nodes[nid].parent_id].append(nid)

        # Score each group by UCB
        group_scores = []
        for pid, children in parent_groups.items():
            signals = [self.nodes[c].intermediate_signal for c in children]
            mu = float(np.mean(signals))
            sigma = float(np.std(signals)) if len(signals) > 1 else 0.0
            ucb = self.alpha1 * mu + self.alpha2 * sigma
            group_scores.append((pid, ucb, children))

        group_scores.sort(key=lambda x: x[1], reverse=True)

        keep = []
        for pid, ucb, children in group_scores:
            if len(keep) >= budget:
                break
            remaining = budget - len(keep)
            if len(children) <= remaining:
                keep.extend(children)
            else:
                # Within group, keep highest-signal children
                children_sorted = sorted(
                    children,
                    key=lambda c: self.nodes[c].intermediate_signal,
                    reverse=True,
                )
                keep.extend(children_sorted[:remaining])

        return keep[:budget]

    def _kill_subtree(self, node_id: int):
        """Mark a node and all its descendants as dead."""
        node = self.nodes[node_id]
        node.is_alive = False
        for child_id in node.children:
            self._kill_subtree(child_id)

    def _reassign_rollout_indices(self):
        """After pruning, reassign contiguous rollout_idx per sample_id."""
        for sid, leaf_ids in self._active_leaves.items():
            alive = [nid for nid in leaf_ids if self.nodes[nid].is_alive]
            for new_idx, nid in enumerate(alive):
                self.nodes[nid].rollout_idx = new_idx

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_leaves(self, sample_id: Optional[str] = None) -> List[TreeNode]:
        """Return all alive leaf nodes, optionally filtered by sample_id."""
        if sample_id is not None:
            return [
                self.nodes[nid]
                for nid in self._active_leaves.get(sample_id, [])
                if self.nodes[nid].is_alive
            ]
        return [
            self.nodes[nid]
            for nids in self._active_leaves.values()
            for nid in nids
            if self.nodes[nid].is_alive
        ]

    def get_grpo_groups(self, session_idx: Optional[int] = None) -> Dict[int, List[TreeNode]]:
        """Return GRPO groups: {parent_id: [child_nodes]}.

        Each group contains siblings (children of the same parent) which
        share the same initial state and can be compared via GRPO.

        Args:
            session_idx: If provided, only return groups at this session level.
                         If None, return groups across all sessions.
        """
        groups: Dict[int, List[TreeNode]] = defaultdict(list)
        for node in self.nodes.values():
            if not node.is_alive:
                continue
            if session_idx is not None and node.session_idx != session_idx:
                continue
            if node.parent_id is not None:
                groups[node.parent_id].append(node)

        # Also include root-level groups (parent_id=None, grouped by sample_id)
        if session_idx is None or session_idx == 0:
            root_groups: Dict[str, List[TreeNode]] = defaultdict(list)
            for node in self.nodes.values():
                if node.is_alive and node.parent_id is None:
                    root_groups[node.sample_id].append(node)
            # Use negative IDs for root groups to avoid collision
            for i, (sid, nodes) in enumerate(root_groups.items()):
                if len(nodes) > 1:
                    groups[-(i + 1)] = nodes

        return dict(groups)

    def get_total_alive_leaves(self) -> int:
        """Return count of all alive leaf nodes."""
        return sum(
            1
            for nids in self._active_leaves.values()
            for nid in nids
            if self.nodes[nid].is_alive
        )

    def get_node_to_parent_map(self) -> Dict[int, Optional[int]]:
        """Return {node_id: parent_id} for all alive nodes."""
        return {
            nid: node.parent_id
            for nid, node in self.nodes.items()
            if node.is_alive
        }

    def get_parent_rollout_idx(self, node_id: int) -> Optional[int]:
        """Get the rollout_idx of a node's parent (for memory loading)."""
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        parent = self.nodes.get(node.parent_id)
        return parent.rollout_idx if parent else None

    def get_leaves_per_sample(self) -> Dict[str, int]:
        """Return {sample_id: n_alive_leaves}."""
        return {
            sid: sum(1 for nid in nids if self.nodes[nid].is_alive)
            for sid, nids in self._active_leaves.items()
        }

    def summary(self) -> str:
        """Human-readable summary of the tree state."""
        total = len(self.nodes)
        alive = sum(1 for n in self.nodes.values() if n.is_alive)
        leaves = self.get_total_alive_leaves()
        groups = self.get_grpo_groups()
        lines = [
            f"RolloutTree: {total} total nodes, {alive} alive, {leaves} active leaves",
            f"  GRPO groups: {len(groups)} (avg size {np.mean([len(g) for g in groups.values()]):.1f})" if groups else "  GRPO groups: 0",
            f"  Leaves per sample: {self.get_leaves_per_sample()}",
            f"  Strategy: {self.pruning_strategy} (α1={self.alpha1}, α2={self.alpha2})",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reward propagation (CRITICAL for tree GRPO)
    # ------------------------------------------------------------------

    def get_ancestor_chain(self, node_id: int) -> List[int]:
        """Trace a node back to its root, returning [node_id, parent_id, ..., root_id]."""
        chain = []
        nid = node_id
        while nid is not None:
            chain.append(nid)
            nid = self.nodes[nid].parent_id if nid in self.nodes else None
        return chain

    def get_terminal_descendants(self, node_id: int) -> List[int]:
        """Find all alive terminal leaf nodes that descend from ``node_id``.

        A terminal leaf is an alive node with no alive children (i.e., it's in
        the final session's active leaves, or it's a pruned branch's last alive node).
        """
        node = self.nodes.get(node_id)
        if node is None or not node.is_alive:
            return []

        alive_children = [cid for cid in node.children if self.nodes[cid].is_alive]
        if not alive_children:
            # This node IS a terminal leaf
            return [node_id]

        # Recurse into alive children
        result = []
        for cid in alive_children:
            result.extend(self.get_terminal_descendants(cid))
        return result

    def build_reward_propagation_map(
        self,
        terminal_reward_map: Dict[Tuple[str, int], float],
    ) -> Dict[Tuple[str, int, int], float]:
        """Build a complete mapping for reward propagation to ALL alive nodes.

        The reward propagation problem: `rollout_idx` is reassigned after pruning,
        so `(sample_id, rollout_idx)` at session 0 doesn't match the same trajectory's
        key at the terminal session. This method resolves it using the tree structure.

        For each alive node, its reward is the **average** reward of its terminal
        descendants (MERS-style credit assignment across branches).

        Args:
            terminal_reward_map: {(sample_id, terminal_rollout_idx): reward}
                Built from the terminal batch using the terminal session's rollout_idx.

        Returns:
            {(sample_id, session_idx, rollout_idx): reward}
                Mapping for ALL alive nodes at ALL session levels.
                The rollout_idx is the node's rollout_idx at its specific session level.
        """
        propagated = {}

        for node in self.nodes.values():
            # If this node already has a true QA reward calculated during an anchor session,
            # we should use that directly instead of propagating from terminal descendants.
            if getattr(self, 'pruning_strategy', '') == 'anchor' and hasattr(node, 'reward') and node.reward != 0.0:
                propagated[(node.sample_id, node.session_idx, node.rollout_idx)] = node.reward
                continue

            # Find this node's terminal descendants
            terminal_ids = self.get_terminal_descendants(node.node_id)
            if not terminal_ids:
                continue

            # Gather terminal rewards
            terminal_rewards = []
            for tid in terminal_ids:
                t_node = self.nodes[tid]
                key = (t_node.sample_id, t_node.rollout_idx)
                if key in terminal_reward_map:
                    terminal_rewards.append(terminal_reward_map[key])

            if terminal_rewards:
                # Average of descendants' rewards
                avg_reward = sum(terminal_rewards) / len(terminal_rewards)
            else:
                avg_reward = 0.0

            propagated[(node.sample_id, node.session_idx, node.rollout_idx)] = avg_reward

        return propagated

    def build_per_session_reward_propagation_map(
        self,
        terminal_per_session_map: Dict[Tuple[str, int], 'torch.Tensor'],
    ) -> Dict[Tuple[str, int, int], 'torch.Tensor']:
        """Like build_reward_propagation_map but for per-session F1 tensors.

        For nodes with multiple terminal descendants, averages across descendants'
        per-session reward vectors.

        Args:
            terminal_per_session_map: {(sample_id, terminal_rollout_idx): per_session_tensor}

        Returns:
            {(sample_id, session_idx, rollout_idx): per_session_tensor}
        """
        propagated = {}

        for node in self.nodes.values():
            terminal_ids = self.get_terminal_descendants(node.node_id)

            terminal_tensors = []
            for tid in terminal_ids:
                t_node = self.nodes[tid]
                key = (t_node.sample_id, t_node.rollout_idx)
                if key in terminal_per_session_map:
                    terminal_tensors.append(terminal_per_session_map[key])

            if terminal_tensors:
                import torch
                avg_tensor = torch.stack(terminal_tensors).mean(dim=0)
            else:
                # No terminal descendants (pruned) -> 0 reward for all sessions
                import torch
                # Create a zero tensor of the same shape as a default terminal tensor
                # We need to peek at one of the items in the map to get the shape
                # This assumes terminal_per_session_map is not empty.
                # If it is empty, this will raise StopIteration, which is fine
                # as it implies no terminal rewards were ever computed.
                first_tensor = next(iter(terminal_per_session_map.values()))
                avg_tensor = torch.zeros_like(first_tensor)

            propagated[(node.sample_id, node.session_idx, node.rollout_idx)] = avg_tensor

        return propagated


# ------------------------------------------------------------------
# Intermediate signal computation
# ------------------------------------------------------------------


def compute_intermediate_signal(
    memory_snapshot,
    session_evidences: list,
    total_sessions_tokens: int,
    alpha_evidence: float = 0.7,
    alpha_compression: float = 0.3,
) -> float:
    """Compute a lightweight pruning signal from memory quality metrics.

    This uses the SAME metrics already computed in ``locomo_score()``
    (lines 217–229 and 176 of rema.py) but WITHOUT calling the expensive
    QA judge LLM.

    Args:
        memory_snapshot: A Memory instance (has .dia_ids_set, .total_tokens, .memories).
        session_evidences: List of evidence dia_ids needed for this session.
        total_sessions_tokens: Total tokens across all sessions (for compression ratio).
        alpha_evidence: Weight for evidence coverage in the combined signal.
        alpha_compression: Weight for compression ratio in the combined signal.

    Returns:
        A scalar signal in [0, 1] representing memory quality.
    """
    evidence_coverage = 0.0
    compression_ratio = 0.0

    if memory_snapshot is not None:
        # Evidence coverage (same logic as locomo_score lines 217-229)
        if session_evidences and hasattr(memory_snapshot, "dia_ids_set"):
            session_evidences_set = set(session_evidences)
            covered = memory_snapshot.dia_ids_set.intersection(session_evidences_set)
            if len(session_evidences_set) > 0:
                evidence_coverage = len(covered) / len(session_evidences_set)

        # Compression ratio (same logic as locomo_score line 176)
        if (
            hasattr(memory_snapshot, "total_tokens")
            and total_sessions_tokens > 0
        ):
            compression_ratio = max(
                0.0, 1.0 - (memory_snapshot.total_tokens / total_sessions_tokens)
            )

    return alpha_evidence * evidence_coverage + alpha_compression * compression_ratio


# ------------------------------------------------------------------
# Memory forking helpers
# ------------------------------------------------------------------


def fork_memory_snapshots(
    parent_rollout_idx: int,
    child_rollout_indices: List[int],
    sample_id: str,
    chunk_id: int,
    epoch: int,
    split: str = "train",
    base_dir: Optional[str] = None,
):
    """Copy a parent's memory snapshot to multiple children.

    After session s, parent rollout p has a cached memory snapshot.
    Before session s+1, each child of p needs to start from p's memory.
    This function copies p's snapshot file to each child's location.

    Args:
        parent_rollout_idx: The rollout_idx (index_in_batch) of the parent.
        child_rollout_indices: List of rollout_idx values for children.
        sample_id: Conversation ID.
        chunk_id: The chunk_id of the parent's snapshot (session completed).
        epoch: Training epoch.
        split: Data split.
        base_dir: Memory cache directory. Auto-detected from env if None.
    """
    import os

    if base_dir is None:
        base_dir = os.environ.get("MEMORY_CACHE_DIR")
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), "memory_snapshots")

    snapshot_dir = Path(base_dir) / f"epoch_{epoch}" / sample_id

    # Determine parent snapshot path
    parent_name = f"chunk_{chunk_id}_idx_{parent_rollout_idx}"
    parent_pkl = snapshot_dir / f"{parent_name}.pkl"
    parent_json = snapshot_dir / f"{parent_name}.json"

    if parent_pkl.exists():
        src = parent_pkl
        ext = ".pkl"
    elif parent_json.exists():
        src = parent_json
        ext = ".json"
    else:
        print(
            f"[fork_memory] WARNING: No parent snapshot at {snapshot_dir}/{parent_name}.*"
        )
        return

    for child_idx in child_rollout_indices:
        if child_idx == parent_rollout_idx:
            continue  # Skip self-copy
        child_name = f"chunk_{chunk_id}_idx_{child_idx}{ext}"
        dst = snapshot_dir / child_name
        shutil.copy2(str(src), str(dst))

    print(
        f"[fork_memory] Forked {sample_id} chunk {chunk_id} "
        f"parent_idx={parent_rollout_idx} → children {child_rollout_indices}"
    )
