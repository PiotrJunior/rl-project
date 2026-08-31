"""Fixed-capacity segment trees backing prioritized replay.

Prioritized replay needs two operations at every insertion and sample:
proportional sampling by priority (a prefix-sum search) and the running minimum
priority (for importance-weight normalisation).  Both are O(log n) here.

Implemented on flat numpy arrays rather than python lists: the sampling path
runs ``batch_size`` prefix-sum searches every training step, and this is the
one non-network hot spot in the training loop.
"""

from __future__ import annotations

import numpy as np


class SegmentTree:
    """Complete binary tree over ``capacity`` leaves, reduced by ``op``."""

    def __init__(self, capacity: int, op: str, neutral: float) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError(f"capacity must be a positive power of two, got {capacity}")
        self.capacity = capacity
        self.neutral = neutral
        self._op_name = op
        self._op = {"sum": np.add, "min": np.minimum}[op]
        self.tree = np.full(2 * capacity, neutral, dtype=np.float64)

    def __setitem__(self, idx: int, value: float) -> None:
        if not 0 <= idx < self.capacity:
            raise IndexError(f"leaf index {idx} out of range [0, {self.capacity})")
        i = idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self._op(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def __getitem__(self, idx: int) -> float:
        if not 0 <= idx < self.capacity:
            raise IndexError(f"leaf index {idx} out of range [0, {self.capacity})")
        return float(self.tree[idx + self.capacity])

    def set_many(self, idxs: np.ndarray, values: np.ndarray) -> None:
        """Vectorised multi-leaf update.

        Updating a whole batch of priorities one leaf at a time costs
        ``batch_size * log(capacity)`` python-level iterations per training
        step.  Here we write all leaves, then rebuild each affected tree level
        in one numpy pass -- ``log(capacity)`` iterations regardless of batch
        size.
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if np.any((idxs < 0) | (idxs >= self.capacity)):
            raise IndexError("leaf index out of range")
        positions = idxs + self.capacity
        self.tree[positions] = values
        parents = np.unique(positions // 2)
        parents = parents[parents >= 1]
        while parents.size:
            self.tree[parents] = self._op(
                self.tree[2 * parents], self.tree[2 * parents + 1]
            )
            parents = np.unique(parents // 2)
            parents = parents[parents >= 1]

    def reduce(self) -> float:
        """Reduction over all leaves (total sum, or global min)."""
        return float(self.tree[1])


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, "sum", 0.0)

    def sum(self) -> float:
        return self.reduce()

    def find_prefixsum_idx(self, prefixsum: np.ndarray) -> np.ndarray:
        """Vectorised inverse-CDF lookup.

        For each ``p`` in ``prefixsum`` returns the smallest leaf ``i`` with
        ``sum(tree[:i+1]) > p``.  Descends all queries in lockstep, so the whole
        batch costs ``log(capacity)`` numpy operations.
        """
        prefixsum = np.atleast_1d(np.asarray(prefixsum, dtype=np.float64)).copy()
        total = self.sum()
        if total <= 0.0:
            raise ValueError("cannot sample from a segment tree with zero total priority")
        # Guard against floating-point drift pushing a query past the total,
        # which would otherwise walk off the end of the tree.
        prefixsum = np.clip(prefixsum, 0.0, np.nextafter(total, 0.0))
        idx = np.ones(prefixsum.shape, dtype=np.int64)
        while idx[0] < self.capacity:
            left = 2 * idx
            left_sum = self.tree[left]
            go_right = prefixsum >= left_sum
            prefixsum = np.where(go_right, prefixsum - left_sum, prefixsum)
            idx = np.where(go_right, left + 1, left)
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, "min", float("inf"))

    def min(self) -> float:
        return self.reduce()
