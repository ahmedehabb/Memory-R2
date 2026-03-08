"""Unit tests for tree_rollout.py and tree GRPO advantage computation."""

import pytest
import torch
import numpy as np
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from verl.rema_trainer.ppo.tree_rollout import (
    TreeNode,
    RolloutTree,
    compute_intermediate_signal,
)
from verl.rema_trainer.ppo.core_algos import compute_tree_grpo_outcome_advantage


# ============================================================
# Tree Structure Tests
# ============================================================

class TestRolloutTreeBasics:
    """Test basic tree construction, branching, and querying."""

    def test_create_roots(self):
        tree = RolloutTree(initial_rollouts=4, branch_factor=2, max_active_leaves=8)
        roots = tree.create_roots(["conv-1", "conv-2"])

        # Should create 4 roots per sample = 8 total
        assert len(roots) == 8
        assert tree.get_total_alive_leaves() == 8

        # Roots should have parent_id=None and session_idx=0
        for root in roots:
            assert root.parent_id is None
            assert root.session_idx == 0

        # Check per-sample counts
        leaves_per_sample = tree.get_leaves_per_sample()
        assert leaves_per_sample["conv-1"] == 4
        assert leaves_per_sample["conv-2"] == 4

    def test_branch(self):
        tree = RolloutTree(initial_rollouts=3, branch_factor=2, max_active_leaves=12)
        tree.create_roots(["conv-1"])

        # Branch from session 0 → session 1
        new_nodes = tree.branch(session_idx=1)

        # 3 parents × 2 children = 6 new nodes
        assert len(new_nodes) == 6
        assert tree.get_total_alive_leaves() == 6

        # All new nodes should reference their parent
        for node in new_nodes:
            assert node.parent_id is not None
            assert node.session_idx == 1
            parent = tree.nodes[node.parent_id]
            assert parent.session_idx == 0

    def test_branch_creates_valid_grpo_groups(self):
        tree = RolloutTree(initial_rollouts=2, branch_factor=3, max_active_leaves=12)
        tree.create_roots(["conv-1"])
        tree.branch(session_idx=1)

        groups = tree.get_grpo_groups(session_idx=1)

        # 2 parents → 2 GRPO groups
        assert len(groups) == 2

        # Each group has 3 siblings
        for pid, siblings in groups.items():
            assert len(siblings) == 3
            # All siblings share same parent
            parent_ids = {s.parent_id for s in siblings}
            assert len(parent_ids) == 1

    def test_multiple_branches(self):
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=16)
        tree.create_roots(["conv-1"])

        # Session 1: 2 → 4
        tree.branch(session_idx=1)
        assert tree.get_total_alive_leaves() == 4

        # Session 2: 4 → 8
        tree.branch(session_idx=2)
        assert tree.get_total_alive_leaves() == 8


class TestRolloutTreePruning:
    """Test pruning strategies."""

    def _setup_tree_with_signals(self):
        """Create a tree and set intermediate signals for testing."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=3, max_active_leaves=4)
        tree.create_roots(["conv-1"])
        new_nodes = tree.branch(session_idx=1)

        # 2 parents × 3 children = 6 nodes
        # Set diverse signals to test pruning
        signals = {}
        for i, node in enumerate(new_nodes):
            # Group 0 (parent 0): signals [0.1, 0.5, 0.9] → high variance
            # Group 1 (parent 1): signals [0.4, 0.45, 0.5] → low variance
            if node.parent_id == 0:
                signals[node.node_id] = [0.1, 0.5, 0.9][i % 3]
            else:
                signals[node.node_id] = [0.4, 0.45, 0.5][i % 3]

        tree.set_intermediate_signals(signals)
        return tree, new_nodes

    def test_prune_respects_budget(self):
        tree, _ = self._setup_tree_with_signals()
        assert tree.get_total_alive_leaves() == 6

        pruned = tree.prune_to_budget(budget=3)
        assert tree.get_total_alive_leaves() <= 3
        assert len(pruned) >= 3  # At least 3 pruned

    def test_variance_pruning_prefers_high_variance_groups(self):
        tree = RolloutTree(
            initial_rollouts=2, branch_factor=3, max_active_leaves=4,
            pruning_strategy="variance"
        )
        tree.create_roots(["conv-1"])
        new_nodes = tree.branch(session_idx=1)

        # Group 0 (high variance): signals = [0.0, 0.5, 1.0]
        # Group 1 (low variance): signals = [0.5, 0.5, 0.5]
        signals = {}
        groups = defaultdict(list)
        for node in new_nodes:
            groups[node.parent_id].append(node)

        group_list = list(groups.items())
        for i, node in enumerate(group_list[0][1]):
            signals[node.node_id] = [0.0, 0.5, 1.0][i]
        for i, node in enumerate(group_list[1][1]):
            signals[node.node_id] = [0.5, 0.5, 0.5][i]

        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=3)

        # After pruning to 3, all 3 should come from the high-variance group
        alive = tree.get_active_leaves()
        alive_parents = {n.parent_id for n in alive}
        assert group_list[0][0] in alive_parents

    def test_quality_pruning_keeps_highest(self):
        tree = RolloutTree(
            initial_rollouts=1, branch_factor=5, max_active_leaves=3,
            pruning_strategy="quality"
        )
        tree.create_roots(["conv-1"])
        new_nodes = tree.branch(session_idx=1)

        signals = {n.node_id: float(i) / 4.0 for i, n in enumerate(new_nodes)}
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=3)

        alive = tree.get_active_leaves()
        alive_signals = [n.intermediate_signal for n in alive]
        # Top 3 should have the highest signals
        assert min(alive_signals) >= 0.4  # 3rd highest = 2/4 = 0.5 or 3/4
    def test_pruned_leaves_keep_original_rollout_indices(self):
        """Verify that pruned nodes do NOT get their rollout_idx reassigned, which we need for memory forking."""
        tree = RolloutTree(
            initial_rollouts=1, branch_factor=5, max_active_leaves=3,
            pruning_strategy="quality"
        )
        tree.create_roots(["conv-1"])
        new_nodes = tree.branch(session_idx=1)
        # new_nodes will have rollout_idx 0, 1, 2, 3, 4
        
        # Kill nodes 0 and 1 by giving them low signals
        signals = {
            new_nodes[0].node_id: 0.1,
            new_nodes[1].node_id: 0.2,
            new_nodes[2].node_id: 0.9,
            new_nodes[3].node_id: 0.8,
            new_nodes[4].node_id: 0.7,
        }
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=3)
        
        # The survivors should be nodes 2, 3, 4. Their rollout_idx should remain 2, 3, 4.
        alive = tree.get_active_leaves()
        alive_rollout_indices = {n.rollout_idx for n in alive}
        assert alive_rollout_indices == {2, 3, 4}

    def test_intra_pruning_balances_children(self):
        """Intra-pruning should guarantee N children per parent to protect GRPO math, rather than picking global top N."""
        # 4 roots, each branching into 4 -> 16 total generated. We want to keep 8.
        tree = RolloutTree(initial_rollouts=4, branch_factor=4, max_active_leaves=8, pruning_strategy="intra")
        roots = tree.create_roots(["conv-1"])
        
        # Branch roots once
        tree.branch(session_idx=1)
            
        leaves = tree.get_active_leaves()
        assert len(leaves) == 16
        
        # Parent 0 is amazing: children score [0.99, 0.98, 0.97, 0.96]
        # Parent 1 is okay: children score [0.80, 0.70, 0.60, 0.50]
        # Parent 2 is bad: children score [0.40, 0.35, 0.30, 0.25]
        # Parent 3 is worst: children score [0.20, 0.15, 0.10, 0.05]
        
        signals = {}
        for leaf in leaves:
            parent_idx = leaf.rollout_idx // 4
            if parent_idx == 0: signals[leaf.node_id] = 0.99 - (leaf.rollout_idx % 4) * 0.01
            elif parent_idx == 1: signals[leaf.node_id] = 0.80 - (leaf.rollout_idx % 4) * 0.10
            elif parent_idx == 2: signals[leaf.node_id] = 0.40 - (leaf.rollout_idx % 4) * 0.05
            elif parent_idx == 3: signals[leaf.node_id] = 0.20 - (leaf.rollout_idx % 4) * 0.05
            
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=8)
        
        survivors = tree.get_active_leaves()
        assert len(survivors) == 8
        
        # IntraP should pick Top 4 Parents (it picked all 4 because we have 4 parents)
        # Budget is 8 -> 8 // 4 = 2 children per parent!
        # It should NOT just grab all 4 from Parent 0 and all 4 from Parent 1.
        parent_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in survivors:
            parent_idx = s.rollout_idx // 4
            parent_counts[parent_idx] += 1
            
        assert parent_counts == {0: 2, 1: 2, 2: 2, 3: 2}, "IntraP failed to evenly balance children across parents"

    def test_inter_pruning_keeps_all_children(self):
        """Inter-pruning should rank parents and keep ALL of their children, abandoning unselected parents."""
        tree = RolloutTree(initial_rollouts=4, branch_factor=4, max_active_leaves=8, pruning_strategy="inter")
        roots = tree.create_roots(["conv-1"])
        tree.branch(session_idx=1)
            
        leaves = tree.get_active_leaves()
        
        # Same scoring as above
        signals = {}
        for leaf in leaves:
            parent_idx = leaf.rollout_idx // 4
            if parent_idx == 0: signals[leaf.node_id] = 0.99 - (leaf.rollout_idx % 4) * 0.01
            elif parent_idx == 1: signals[leaf.node_id] = 0.80 - (leaf.rollout_idx % 4) * 0.10
            elif parent_idx == 2: signals[leaf.node_id] = 0.40 - (leaf.rollout_idx % 4) * 0.05
            elif parent_idx == 3: signals[leaf.node_id] = 0.20 - (leaf.rollout_idx % 4) * 0.05
            
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=8)
        
        survivors = tree.get_active_leaves()
        assert len(survivors) == 8
        
        # InterP should pick Top 2 Parents (Parent 0 and Parent 1) and keep ALL 4 of their children
        parent_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in survivors:
            parent_idx = s.rollout_idx // 4
            parent_counts[parent_idx] += 1
            
        assert parent_counts == {0: 4, 1: 4, 2: 0, 3: 0}, "InterP failed to group all children under winning parents"


class TestGRPOGroups:
    """Test GRPO group extraction."""

    def test_root_level_groups(self):
        tree = RolloutTree(initial_rollouts=4, branch_factor=2, max_active_leaves=8)
        tree.create_roots(["conv-1"])

        groups = tree.get_grpo_groups(session_idx=0)
        # Root level: 4 roots for conv-1 → 1 group with 4 members
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert len(group) == 4

    def test_multi_sample_groups(self):
        tree = RolloutTree(initial_rollouts=3, branch_factor=2, max_active_leaves=12)
        tree.create_roots(["conv-1", "conv-2"])
        tree.branch(session_idx=1)

        groups = tree.get_grpo_groups(session_idx=1)
        # 3 parents per sample × 2 samples = 6 groups
        assert len(groups) == 6
        for pid, siblings in groups.items():
            assert len(siblings) == 2  # branch_factor=2


# ============================================================
# Tree GRPO Advantage Tests
# ============================================================

class TestTreeGRPOAdvantage:
    """Test the tree GRPO advantage computation."""

    def test_sibling_normalization(self):
        """Siblings should have zero-mean advantages within their group."""
        token_rewards = torch.zeros(4, 10)
        # Two groups of 2 siblings each
        # Group (parent=0): rewards 1.0, 3.0 → normalized ≈ -1.0, 1.0
        # Group (parent=1): rewards 2.0, 2.0 → normalized = 0.0, 0.0
        token_rewards[0, -1] = 1.0  # parent=0, child 1
        token_rewards[1, -1] = 3.0  # parent=0, child 2
        token_rewards[2, -1] = 2.0  # parent=1, child 1
        token_rewards[3, -1] = 2.0  # parent=1, child 2

        eos_mask = torch.ones(4, 10)
        parent_id = torch.tensor([0, 0, 1, 1])

        advantages, returns = compute_tree_grpo_outcome_advantage(
            token_rewards, eos_mask, parent_id
        )

        # Group 0: should have opposite-sign advantages
        assert advantages[0, -1].item() * advantages[1, -1].item() < 0  # opposite signs

        # Group 1: identical rewards → zero advantage
        assert abs(advantages[2, -1].item()) < 1e-5
        assert abs(advantages[3, -1].item()) < 1e-5

    def test_single_child_gets_zero(self):
        """A single child in a group should get zero advantage."""
        token_rewards = torch.zeros(3, 5)
        token_rewards[0, -1] = 5.0  # parent=0, only child
        token_rewards[1, -1] = 2.0  # parent=1, child 1
        token_rewards[2, -1] = 4.0  # parent=1, child 2

        eos_mask = torch.ones(3, 5)
        parent_id = torch.tensor([0, 1, 1])

        advantages, _ = compute_tree_grpo_outcome_advantage(
            token_rewards, eos_mask, parent_id
        )

        # Single child → zero everywhere
        assert (advantages[0] == 0).all()

        # Group 1: non-zero advantages
        assert advantages[1, -1].item() != 0
        assert advantages[2, -1].item() != 0

    def test_eos_mask_applied(self):
        """Advantages should be zero where eos_mask is zero."""
        token_rewards = torch.zeros(2, 8)
        token_rewards[0, -1] = 1.0
        token_rewards[1, -1] = 3.0

        eos_mask = torch.zeros(2, 8)
        eos_mask[:, 3:6] = 1  # Only positions 3-5 active
        parent_id = torch.tensor([0, 0])

        advantages, _ = compute_tree_grpo_outcome_advantage(
            token_rewards, eos_mask, parent_id
        )

        # Non-mask positions should be zero
        assert (advantages[:, :3] == 0).all()
        assert (advantages[:, 6:] == 0).all()


# ============================================================
# Intermediate Signal Tests
# ============================================================

class TestIntermediateSignal:
    """Test the lightweight intermediate signal computation."""

    def test_full_coverage_full_compression(self):
        """Perfect memory → signal = 1.0."""

        class MockMemory:
            dia_ids_set = {"D1:1", "D1:2", "D2:1"}
            total_tokens = 10

        signal = compute_intermediate_signal(
            MockMemory(),
            session_evidences=["D1:1", "D1:2", "D2:1"],
            total_sessions_tokens=100,  # compression = 1 - 10/100 = 0.9
            alpha_evidence=0.5,
            alpha_compression=0.5,
        )
        # 0.5 * 1.0 + 0.5 * 0.9 = 0.95
        assert abs(signal - 0.95) < 1e-6

    def test_no_memory(self):
        """No memory → signal = 0.0."""
        signal = compute_intermediate_signal(
            None,
            session_evidences=["D1:1"],
            total_sessions_tokens=100,
        )
        assert signal == 0.0

    def test_partial_coverage(self):
        """Partial evidence coverage."""

        class MockMemory:
            dia_ids_set = {"D1:1"}
            total_tokens = 50

        signal = compute_intermediate_signal(
            MockMemory(),
            session_evidences=["D1:1", "D1:2"],
            total_sessions_tokens=100,
            alpha_evidence=1.0,
            alpha_compression=0.0,
        )
        # evidence_coverage = 1/2 = 0.5
        assert abs(signal - 0.5) < 1e-6


# ============================================================
# Integration-like test
# ============================================================

class TestTreeWorkflow:
    """End-to-end tree workflow test."""

    def test_full_3_session_workflow(self):
        """Simulate 3 sessions with branching and pruning."""
        tree = RolloutTree(
            initial_rollouts=4,
            branch_factor=2,
            max_active_leaves=8,
            pruning_strategy="quality",
        )

        # Session 0: Create roots
        roots = tree.create_roots(["conv-1", "conv-2"])
        assert tree.get_total_alive_leaves() == 8  # 4 per sample

        # Session 1: Branch (4 parents → 8 children per sample = 16 total)
        children_s1 = tree.branch(session_idx=1)
        assert tree.get_total_alive_leaves() == 16

        # Set signals and prune to 8 (4 per sample)
        signals = {n.node_id: np.random.random() for n in children_s1}
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=4)
        assert tree.get_total_alive_leaves() <= 8

        # Session 2: Branch again
        children_s2 = tree.branch(session_idx=2)
        assert tree.get_total_alive_leaves() > 0

        # Set final rewards and check GRPO groups
        rewards = {n.node_id: np.random.random() for n in children_s2}
        tree.set_rewards(rewards)

        groups = tree.get_grpo_groups(session_idx=2)
        assert len(groups) > 0

        # Verify siblings share same parent
        for pid, siblings in groups.items():
            if pid >= 0:  # Skip root pseudo-groups
                parent_ids = {s.parent_id for s in siblings}
                assert len(parent_ids) == 1
                assert list(parent_ids)[0] == pid

        print(tree.summary())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================
# Reward Propagation Tests (CRITICAL)
# ============================================================

class TestRewardPropagation:
    """Test that rewards correctly propagate through the tree despite rollout_idx reassignment."""

    def test_ancestor_chain(self):
        """Test tracing a node back to root."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=4)
        tree.create_roots(["conv-1"])
        children = tree.branch(session_idx=1)

        # Pick a child and trace back
        child = children[0]
        chain = tree.get_ancestor_chain(child.node_id)

        # Should be [child_node_id, parent_node_id]
        assert len(chain) == 2
        assert chain[0] == child.node_id
        assert chain[1] == child.parent_id

    def test_terminal_descendants(self):
        """Test finding terminal leaves of an intermediate node."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=8)
        tree.create_roots(["conv-1"])
        tree.branch(session_idx=1)  # 2→4
        tree.branch(session_idx=2)  # 4→8

        # Root node 0 should have 4 terminal descendants (2 children × 2 grandchildren)
        root_0_descendants = tree.get_terminal_descendants(0)
        assert len(root_0_descendants) == 4

    def test_terminal_descendants_after_pruning(self):
        """After pruning, dead branches should NOT appear as descendants."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=4,
                          pruning_strategy="quality")
        tree.create_roots(["conv-1"])
        children = tree.branch(session_idx=1)  # 2→4

        # Give different signals: high to first 2, low to last 2
        signals = {}
        for i, child in enumerate(children):
            signals[child.node_id] = 1.0 if i < 2 else 0.0
        tree.set_intermediate_signals(signals)
        tree.prune_to_budget(budget=2)

        # Root 0 should have at most 2 terminal descendants (pruned some)
        root_0_desc = tree.get_terminal_descendants(0)
        assert len(root_0_desc) <= 2

    def test_reward_propagation_map_simple(self):
        """Test basic reward propagation through tree."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=4)
        tree.create_roots(["conv-1"])
        children = tree.branch(session_idx=1)

        # Terminal rewards at session 1 (terminal session)
        terminal_reward_map = {}
        for child in children:
            terminal_reward_map[(child.sample_id, child.rollout_idx)] = float(child.rollout_idx)

        prop_map = tree.build_reward_propagation_map(terminal_reward_map)

        # Session 1 nodes should get their own reward
        for child in children:
            key = ("conv-1", 1, child.rollout_idx)
            assert key in prop_map
            assert prop_map[key] == float(child.rollout_idx)

        # Session 0 roots should get the AVERAGE of their children's rewards
        root_0_key = ("conv-1", 0, 0)  # root 0
        root_1_key = ("conv-1", 0, 1)  # root 1
        assert root_0_key in prop_map
        assert root_1_key in prop_map

        # Root 0 children: rollout_idx 0 and 1 (reward 0.0 and 1.0) → avg = 0.5
        assert abs(prop_map[root_0_key] - 0.5) < 1e-6

    def test_reward_propagation_after_pruning(self):
        """CRITICAL: Test that reward propagation works after rollout_idx reassignment."""
        tree = RolloutTree(initial_rollouts=4, branch_factor=2, max_active_leaves=4,
                          pruning_strategy="quality")
        tree.create_roots(["conv-1"])  # 4 roots with rollout_idx 0,1,2,3
        children = tree.branch(session_idx=1)  # 4×2=8 children

        # Set signals: children 0-3 high, children 4-7 low
        signals = {}
        for i, child in enumerate(children):
            signals[child.node_id] = 1.0 - (i * 0.1)
        tree.set_intermediate_signals(signals)

        # Prune to 4 → rollout_idx gets reassigned to 0,1,2,3
        tree.prune_to_budget(budget=4)
        alive_leaves = tree.get_active_leaves()
        assert len(alive_leaves) == 4

        # After reassignment, rollout_idx should be 0,1,2,3
        rollout_idxs = sorted([n.rollout_idx for n in alive_leaves])
        assert rollout_idxs == [0, 1, 2, 3]

        # Set terminal rewards using the REASSIGNED rollout_idx
        terminal_reward_map = {}
        for leaf in alive_leaves:
            terminal_reward_map[(leaf.sample_id, leaf.rollout_idx)] = leaf.rollout_idx * 10.0

        prop_map = tree.build_reward_propagation_map(terminal_reward_map)

        # Verify session 1 entries exist with correct rewards
        for leaf in alive_leaves:
            key = ("conv-1", 1, leaf.rollout_idx)
            assert key in prop_map
            assert prop_map[key] == leaf.rollout_idx * 10.0

        # Verify session 0 entries exist (roots may have only 1 descendant after pruning)
        session_0_entries = {k: v for k, v in prop_map.items() if k[1] == 0}
        assert len(session_0_entries) > 0
        for k, v in session_0_entries.items():
            assert v >= 0  # Should have a valid reward

    def test_per_session_reward_propagation(self):
        """Test per-session F1 tensor propagation through tree."""
        tree = RolloutTree(initial_rollouts=2, branch_factor=2, max_active_leaves=4)
        tree.create_roots(["conv-1"])
        children = tree.branch(session_idx=1)

        # Per-session rewards: [session_0_f1, session_1_f1]
        terminal_per_session_map = {}
        for child in children:
            f1 = torch.tensor([0.5 + child.rollout_idx * 0.1, 0.3 + child.rollout_idx * 0.1])
            terminal_per_session_map[(child.sample_id, child.rollout_idx)] = f1

        prop_map = tree.build_per_session_reward_propagation_map(terminal_per_session_map)

        # Session 0 roots should get averaged per-session tensors
        root_0_key = ("conv-1", 0, 0)
        assert root_0_key in prop_map
        assert prop_map[root_0_key].shape == (2,)


class TestRayTrainerRewardFlow:
    """Test simulating the ray_trainer reward propagation flow."""

    def test_full_reward_flow_3_sessions(self):
        """Simulate 3 sessions with branching+pruning and verify reward propagation."""
        tree = RolloutTree(initial_rollouts=4, branch_factor=2, max_active_leaves=4,
                          pruning_strategy="quality")

        # Session 0: 4 rollouts
        tree.create_roots(["conv-A"])
        session_0_nodes = tree.get_active_leaves()
        assert len(session_0_nodes) == 4

        # Session 1: branch 4→8, prune to 4
        children_s1 = tree.branch(session_idx=1)
        signals_s1 = {n.node_id: np.random.random() for n in children_s1}
        tree.set_intermediate_signals(signals_s1)
        tree.prune_to_budget(budget=4)
        session_1_nodes = tree.get_active_leaves()
        assert len(session_1_nodes) == 4

        # Session 2: branch 4→8, prune to 4
        children_s2 = tree.branch(session_idx=2)
        signals_s2 = {n.node_id: np.random.random() for n in children_s2}
        tree.set_intermediate_signals(signals_s2)
        tree.prune_to_budget(budget=4)
        terminal_nodes = tree.get_active_leaves()
        assert len(terminal_nodes) == 4

        # Simulate terminal rewards (like the reward function would produce)
        global_reward_map = {}
        for term_node in terminal_nodes:
            global_reward_map[(term_node.sample_id, term_node.rollout_idx)] = np.random.random()

        # Build tree-aware reward map
        tree_reward_map = tree.build_reward_propagation_map(global_reward_map)

        # Verify every alive node at every session has a reward entry
        for node in tree.nodes.values():
            if not node.is_alive:
                continue
            key = (node.sample_id, node.session_idx, node.rollout_idx)
            assert key in tree_reward_map, f"Missing reward for node {node.node_id} at session {node.session_idx}"

        # Verify terminal nodes get their exact reward
        for term_node in terminal_nodes:
            tree_key = (term_node.sample_id, 2, term_node.rollout_idx)
            flat_key = (term_node.sample_id, term_node.rollout_idx)
            assert abs(tree_reward_map[tree_key] - global_reward_map[flat_key]) < 1e-6

        # Verify intermediate nodes get averaged rewards
        for node in tree.nodes.values():
            if not node.is_alive or node.session_idx == 2:
                continue
            key = (node.sample_id, node.session_idx, node.rollout_idx)
            descendants = tree.get_terminal_descendants(node.node_id)
            expected_rewards = []
            for d in descendants:
                d_node = tree.nodes[d]
                d_key = (d_node.sample_id, d_node.rollout_idx)
                if d_key in global_reward_map:
                    expected_rewards.append(global_reward_map[d_key])
            if expected_rewards:
                expected_avg = sum(expected_rewards) / len(expected_rewards)
                assert abs(tree_reward_map[key] - expected_avg) < 1e-6, (
                    f"Node {node.node_id} at session {node.session_idx}: "
                    f"expected {expected_avg}, got {tree_reward_map[key]}"
                )


class TestTreeGrowth:
    """Test the tree growth scenario where initial_rollouts < max_active_leaves."""

    def test_growth_phase_creates_meaningful_groups(self):
        """Start with 4 rollouts, grow to 16 via branch_factor=4.
        
        Growth phase should create groups of 4 siblings (meaningful GRPO signal).
        """
        tree = RolloutTree(
            initial_rollouts=4,
            branch_factor=4,
            max_active_leaves=16,
            pruning_strategy="quality",
        )

        # Session 0: 4 roots
        roots = tree.create_roots(["conv-1"])
        assert tree.get_total_alive_leaves() == 4
        assert tree.get_leaves_per_sample()["conv-1"] == 4

        # Session 1: Branch 4 × 4 = 16. Budget=16 → all survive (growth phase!)
        children_s1 = tree.branch(session_idx=1)
        assert len(children_s1) == 16
        assert tree.get_total_alive_leaves() == 16

        # No pruning needed — we're at budget
        signals = {n.node_id: np.random.random() for n in children_s1}
        tree.set_intermediate_signals(signals)
        pruned = tree.prune_to_budget()
        assert len(pruned) == 0  # Nothing pruned — still growing!
        assert tree.get_leaves_per_sample()["conv-1"] == 16

        # Check GRPO groups: 4 parents → 4 groups of 4 siblings each
        groups = tree.get_grpo_groups(session_idx=1)
        assert len(groups) == 4
        for pid, siblings in groups.items():
            assert len(siblings) == 4, f"Expected 4 siblings, got {len(siblings)}"

        # Session 2: Branch 16 × 4 = 64. Budget=16 → prune to 16 (steady state)
        children_s2 = tree.branch(session_idx=2)
        assert len(children_s2) == 64
        signals_s2 = {n.node_id: np.random.random() for n in children_s2}
        tree.set_intermediate_signals(signals_s2)
        tree.prune_to_budget()
        assert tree.get_total_alive_leaves() == 16
        assert tree.get_leaves_per_sample()["conv-1"] == 16

    def test_growth_with_multiple_samples(self):
        """Growth with 2 samples: each starts with 4, grows to 8."""
        tree = RolloutTree(
            initial_rollouts=4,
            branch_factor=2,
            max_active_leaves=16,  # 8 per sample
            pruning_strategy="variance",
        )

        roots = tree.create_roots(["conv-A", "conv-B"])
        assert tree.get_total_alive_leaves() == 8  # 4 per sample

        # Branch: 4×2=8 per sample = 16 total. Budget=16 → no pruning
        children = tree.branch(session_idx=1)
        assert tree.get_total_alive_leaves() == 16
        leaves = tree.get_leaves_per_sample()
        assert leaves["conv-A"] == 8
        assert leaves["conv-B"] == 8

        # Verify groups per sample
        groups = tree.get_grpo_groups(session_idx=1)
        # 4 parents per sample × 2 samples = 8 groups of 2 siblings
        assert len(groups) == 8
        for pid, siblings in groups.items():
            assert len(siblings) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
