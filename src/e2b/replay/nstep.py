"""N-step return accumulation.

Sits between the environment loop and the replay buffer: it consumes 1-step
transitions and emits n-step ones, where the emitted transition carries the
*actual* discount applied, ``gamma ** k``.  ``k`` is normally ``n`` but is
shorter for the transitions flushed at the end of an episode, so the agent must
not assume a fixed ``gamma ** n`` in the bootstrap term.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class NStepTransition:
    obs: np.ndarray
    action: int
    reward: float          # sum_{i<k} gamma^i * r_i
    next_obs: np.ndarray   # state reached after k steps
    terminated: bool       # whether that k-step lookahead ended in a true terminal
    discount: float        # gamma ** k, the coefficient on the bootstrap term


class NStepAccumulator:
    """Rolling buffer producing n-step transitions from a 1-step stream."""

    def __init__(self, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        self.n_step = n_step
        self.gamma = gamma
        self._buf: deque[tuple[Any, ...]] = deque(maxlen=n_step)

    def __len__(self) -> int:
        return len(self._buf)

    def _build(self, upto: int) -> NStepTransition:
        """Fold the first ``upto`` buffered 1-step transitions into one.

        The fold stops early at a true terminal, so a transition never
        accumulates reward across an episode boundary.
        """
        obs, action = self._buf[0][0], self._buf[0][1]
        reward = 0.0
        discount = 1.0
        next_obs = self._buf[0][3]
        terminated = False
        for i in range(upto):
            _, _, r, n_obs, term = self._buf[i]
            reward += discount * r
            discount *= self.gamma
            next_obs = n_obs
            if term:
                terminated = True
                break
        return NStepTransition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            discount=discount,
        )

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> Iterator[NStepTransition]:
        """Add a 1-step transition; yield any n-step transitions now complete.

        On episode end (terminated *or* truncated) the buffer is drained, so the
        final few transitions of an episode are emitted with k < n rather than
        being discarded.  Discarding them biases short episodes out of the
        replay distribution, which matters a lot early in training when
        episodes are short.
        """
        self._buf.append((obs, action, float(reward), next_obs, bool(terminated)))

        if len(self._buf) == self.n_step:
            yield self._build(self.n_step)

        if terminated or truncated:
            # Drain: emit progressively shorter lookaheads from every remaining
            # start position. If the buffer was already full we just emitted the
            # full-length transition starting at position 0, so drop it first to
            # avoid emitting it twice; otherwise position 0 has not been emitted
            # yet and must be included.
            if len(self._buf) == self.n_step:
                self._buf.popleft()
            while self._buf:
                yield self._build(len(self._buf))
                self._buf.popleft()
            self._buf.clear()

    def reset(self) -> None:
        self._buf.clear()
