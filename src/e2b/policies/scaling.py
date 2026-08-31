"""Q-value scale normalisation -- the detail that decides whether a Boltzmann
temperature means anything at all.

A temperature is only interpretable relative to the spread of the Q-values it
divides.  That spread is not a constant:

* across environments it differs by orders of magnitude (CartPole returns are
  ~10-500, LunarLander ~-400-300, Atari clipped returns are ~0-50);
* within a single run it grows steadily as the value function inflates from its
  near-zero initialisation towards the true return scale.

So a fixed temperature that produces healthy exploration at 20k steps is
effectively greedy at 300k steps, through no change in the schedule.  Any
comparison of "epsilon-greedy vs Boltzmann" that does not address this is really
measuring an uncontrolled, environment-specific annealing schedule.

Three modes are provided, and the difference between the last two is a genuine
experimental finding rather than plumbing -- see ``report/REPORT.md``.
"""

from __future__ import annotations

import numpy as np

from e2b.utils.running_stats import EMA

_EPS = 1e-8


class QScaler:
    """Maps raw Q-values to the values actually divided by the temperature.

    Modes
    -----
    ``none``
        Pass through (mean-centred, which softmax is invariant to). The
        temperature carries raw Q units. Included so the report can *show* the
        failure mode rather than assert it.

    ``per_state``
        ``(q - mean_a q) / (std_a q + eps)``, computed independently at every
        state. Makes the temperature dimensionless, but destroys information:
        a state where every action is genuinely equally good gets its tiny Q
        spread inflated to unit variance, so the policy becomes *confidently*
        peaked on what is really numerical noise. Exactly backwards from what
        exploration should do.

    ``running``
        ``(q - mean_a q) / sigma_hat``, where ``sigma_hat`` is an EMA of the
        across-action Q spread measured over recently visited states. Also
        dimensionless, but the normaliser is shared across states, so a flat
        Q-row stays flat and yields a near-uniform policy while a peaked row
        stays peaked. This is the default.
    """

    MODES = ("none", "per_state", "running")

    def __init__(self, mode: str = "running", decay: float = 0.999) -> None:
        if mode not in self.MODES:
            raise KeyError(f"unknown q_scaling mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self._spread = EMA(decay=decay)

    def observe(self, q: np.ndarray) -> None:
        """Update the running scale from an acting-time Q-row.

        Only called on the behaviour path, so the scale reflects the state
        distribution the policy is actually exploring in.
        """
        if self.mode == "running":
            self._spread.update(float(np.std(np.asarray(q, dtype=np.float64))))

    @property
    def scale(self) -> float:
        """Current normalising scale (1.0 for modes that do not use one)."""
        if self.mode != "running":
            return 1.0
        if not self._spread.initialised:
            return 1.0
        return max(self._spread.value, _EPS)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        centred = q - q.mean()
        if self.mode == "none":
            return centred
        if self.mode == "per_state":
            return centred / (q.std() + _EPS)
        return centred / self.scale

    def state_dict(self) -> dict:
        return {"mode": self.mode, "spread": self._spread.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.mode = state["mode"]
        self._spread.load_state_dict(state["spread"])
