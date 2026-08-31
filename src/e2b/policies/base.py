"""Exploration-policy interface.

A policy turns a Q-row into a distribution over actions.  It is given the global
environment step (so it can drive its schedules) and, optionally, a measure of
how uncertain the Q-estimate is at this state (used only by the
uncertainty-gated variant, ignored by the others).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from e2b.policies.core import entropy, non_greedy_mass


class ExplorationPolicy(ABC):
    """Base class for every exploration strategy in the study."""

    name: str = "base"

    def __init__(self, num_actions: int) -> None:
        if num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {num_actions}")
        self.num_actions = num_actions
        self._last: dict[str, Any] = {}

    @abstractmethod
    def action_probs(
        self, q: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        """Distribution over actions given a Q-row. Must sum to 1."""

    def observe(self, q: np.ndarray) -> None:
        """Hook for updating running statistics from the acting Q-row."""

    def act(
        self,
        q: np.ndarray,
        step: int,
        rng: np.random.Generator,
        uncertainty: float | None = None,
    ) -> int:
        """Sample an action and record diagnostics for this step."""
        q = np.asarray(q, dtype=np.float64)
        self.observe(q)
        probs = self.action_probs(q, step, uncertainty)
        action = int(rng.choice(self.num_actions, p=probs))
        self._last.update(
            entropy=entropy(probs),
            non_greedy=non_greedy_mass(probs, q),
            q_spread=float(q.std()),
            q_mean=float(q.mean()),
        )
        return action

    def diagnostics(self) -> dict[str, Any]:
        """Knob values and behaviour statistics from the most recent ``act``.

        This is what lets the report distinguish "the schedule moved" from "the
        behaviour changed" -- the two are not the same thing once Q-value scale
        drift is in play.
        """
        return dict(self._last)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None
