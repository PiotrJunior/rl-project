"""Uniform and prioritized replay buffers.

Both store observations in their native dtype (float32 for state vectors, uint8
for Atari frames) in preallocated numpy arrays; nothing is allocated on the
sampling path except the output batch.

The bootstrap-mask column is always present but is all-ones when
``num_heads == 1``, so the ensemble arm and the single-head arms run through
exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from e2b.replay.segment_tree import MinSegmentTree, SumSegmentTree


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    discounts: np.ndarray      # gamma ** k for each sampled transition
    masks: np.ndarray          # (batch, num_heads) bootstrap masks
    weights: np.ndarray        # importance-sampling weights (ones if uniform)
    indices: np.ndarray        # buffer positions, for priority updates

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class ReplayBuffer:
    """Circular uniform replay buffer."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        obs_dtype: np.dtype | type = np.float32,
        num_heads: int = 1,
        bootstrap_prob: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.num_heads = num_heads
        self.bootstrap_prob = bootstrap_prob
        self.rng = rng or np.random.default_rng()

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.masks = np.ones((self.capacity, num_heads), dtype=np.float32)

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def _draw_mask(self) -> np.ndarray:
        if self.num_heads == 1 or self.bootstrap_prob >= 1.0:
            return np.ones(self.num_heads, dtype=np.float32)
        mask = (
            self.rng.random(self.num_heads) < self.bootstrap_prob
        ).astype(np.float32)
        # A transition masked out for every head is dead weight; force one head
        # to keep it so the effective buffer size stays equal to the nominal one.
        if not mask.any():
            mask[self.rng.integers(self.num_heads)] = 1.0
        return mask

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        discount: float,
    ) -> int:
        idx = self._pos
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminated[idx] = float(terminated)
        self.discounts[idx] = discount
        self.masks[idx] = self._draw_mask()
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx

    def _gather(self, indices: np.ndarray, weights: np.ndarray) -> Batch:
        return Batch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            terminated=self.terminated[indices],
            discounts=self.discounts[indices],
            masks=self.masks[indices],
            weights=weights,
            indices=indices,
        )

    def sample(self, batch_size: int, beta: float = 1.0) -> Batch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self.rng.integers(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._gather(indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """No-op for uniform replay; present so callers need no branching."""


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay (Schaul et al. 2016).

    New transitions enter at the current maximum priority so that every
    transition is sampled at least once before its priority is ever based on a
    real TD error.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        obs_dtype: np.dtype | type = np.float32,
        num_heads: int = 1,
        bootstrap_prob: float = 1.0,
        alpha: float = 0.5,
        priority_eps: float = 1e-6,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            capacity, obs_shape, obs_dtype, num_heads, bootstrap_prob, rng
        )
        self.alpha = alpha
        self.priority_eps = priority_eps
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2
        self._sum_tree = SumSegmentTree(tree_capacity)
        self._min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs) -> int:
        idx = super().add(*args, **kwargs)
        p = self._max_priority ** self.alpha
        self._sum_tree[idx] = p
        self._min_tree[idx] = p
        return idx

    def sample(self, batch_size: int, beta: float = 1.0) -> Batch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        total = self._sum_tree.sum()
        # Stratified sampling: one draw per equal-probability stratum. Lower
        # variance than `batch_size` iid draws, and it guarantees the batch
        # spans the whole priority range instead of clustering on a few
        # high-priority transitions.
        edges = np.linspace(0.0, total, batch_size + 1)
        samples = self.rng.uniform(edges[:-1], edges[1:])
        indices = self._sum_tree.find_prefixsum_idx(samples)
        # Guard against the tree returning a slot that has not been written yet
        # (possible only through float drift near the total).
        indices = np.clip(indices, 0, self._size - 1)

        probs = self._sum_tree.tree[indices + self._sum_tree.capacity] / total
        min_prob = self._min_tree.min() / total
        # Normalising by the max possible weight (attained at min_prob) keeps
        # weights in (0, 1] so they only ever scale gradients down.
        max_weight = (min_prob * self._size) ** (-beta)
        weights = ((probs * self._size) ** (-beta) / max_weight).astype(np.float32)
        return self._gather(indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(np.asarray(priorities, dtype=np.float64)) + self.priority_eps
        if np.any(priorities <= 0.0):
            raise ValueError("priorities must be positive")
        self._max_priority = max(self._max_priority, float(priorities.max()))
        scaled = priorities ** self.alpha
        self._sum_tree.set_many(indices, scaled)
        self._min_tree.set_many(indices, scaled)


def build_replay(
    cfg,
    obs_shape: tuple[int, ...],
    obs_dtype: np.dtype | type,
    num_heads: int,
    bootstrap_prob: float,
    rng: np.random.Generator,
) -> ReplayBuffer:
    """Construct the replay buffer described by a :class:`ReplayConfig`.

    ``bootstrap_prob`` comes from the *net* config (it is a property of the
    ensemble, not of the buffer) and is forced to 1.0 for single-head runs.
    """
    prob = bootstrap_prob if num_heads > 1 else 1.0
    if cfg.prioritized:
        return PrioritizedReplayBuffer(
            capacity=cfg.capacity,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            num_heads=num_heads,
            bootstrap_prob=prob,
            alpha=cfg.alpha,
            priority_eps=cfg.priority_eps,
            rng=rng,
        )
    return ReplayBuffer(
        capacity=cfg.capacity,
        obs_shape=obs_shape,
        obs_dtype=obs_dtype,
        num_heads=num_heads,
        bootstrap_prob=prob,
        rng=rng,
    )
