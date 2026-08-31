"""Uncertainty-gated handover from epsilon-greedy to Boltzmann.

The project extension: instead of handing over on a fixed step-count schedule,
hand over when the Q-function has actually earned it.

Given a confidence ``c in [0, 1]`` (see :mod:`e2b.uncertainty.estimators`), the
three knobs of the unified family are interpolated between an
"uncertain" endpoint and a "confident" endpoint::

    eps = eps_uncertain + c * (eps_confident - eps_uncertain)
    tau = geometric_interp(tau_uncertain, tau_confident, c)
    k   = round(k_uncertain + c * (k_confident - k_uncertain))

With the default endpoints (``eps: 1 -> 0.01``, ``k: 1 -> |A|``) confidence 0 is
exactly epsilon-greedy with a high epsilon, and confidence 1 is essentially pure
Boltzmann.  The agent therefore *chooses its own* position on the epsilon-greedy
-> Boltzmann path, per state when the ensemble signal is used.

Two guards matter in practice:

* **Warm-up.** Before the reference quantile has seen enough data, confidence is
  pinned to 0 (epsilon-greedy). Otherwise the very first states -- where the
  estimator has no reference to normalise against -- would report spurious high
  confidence and start the run in near-greedy Boltzmann on a randomly
  initialised network, which is the worst possible combination.
* **Smoothing.** The per-state signal is noisy; an EMA over the confidence
  keeps the *global* trend legible for the diagnostics while the raw per-state
  value still drives the action. ``confidence_smoothing`` controls how much of
  the per-state signal survives -- at 1.0 the gating is fully per-state, at 0.0
  it is fully global (a useful ablation isolating "adaptive timing" from
  "per-state resolution").
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from e2b.policies.base import ExplorationPolicy
from e2b.policies.core import compose_probs, top_k_support
from e2b.policies.scaling import QScaler
from e2b.uncertainty.estimators import UncertaintyEstimator, build_estimator
from e2b.utils.running_stats import EMA


class UncertaintyGatedPolicy(ExplorationPolicy):
    """Exploration whose position on the epsilon-greedy/Boltzmann path is
    driven by measured Q-uncertainty rather than by the step counter."""

    name = "uncertainty_gated"

    def __init__(
        self,
        num_actions: int,
        signal: str = "ensemble",
        eps_uncertain: float = 1.0,
        eps_confident: float = 0.01,
        tau_uncertain: float = 1.0,
        tau_confident: float = 0.2,
        k_uncertain: int = 1,
        k_confident: int | None = None,
        q_scaling: str = "running",
        q_scaling_decay: float = 0.999,
        warmup_steps: int = 5_000,
        confidence_smoothing: float = 1.0,
        confidence_decay: float = 0.99,
        normalise_uncertainty: bool = False,
        estimator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(num_actions)
        self.estimator: UncertaintyEstimator = build_estimator(
            signal, **(estimator_kwargs or {})
        )
        self.eps_uncertain = eps_uncertain
        self.eps_confident = eps_confident
        if tau_uncertain <= 0.0 or tau_confident <= 0.0:
            raise ValueError("temperatures must be strictly positive")
        self.tau_uncertain = tau_uncertain
        self.tau_confident = tau_confident
        self.k_uncertain = k_uncertain
        self.k_confident = num_actions if k_confident is None else k_confident
        self.scaler = QScaler(q_scaling, decay=q_scaling_decay)
        # Feed the policy's own running Q-scale to the estimator so the
        # uncertainty signal is dimensionless. See
        # UncertaintyEstimator.set_scale_provider for why this matters; without
        # it the signal grows with the Q-values and the gate never fires.
        self.normalise_uncertainty = bool(normalise_uncertainty)
        if self.normalise_uncertainty:
            if q_scaling != "running":
                # Every other mode returns a constant scale of 1.0, so this
                # would silently be a no-op rather than an error.
                raise ValueError(
                    "normalise_uncertainty requires q_scaling='running'; "
                    f"got {q_scaling!r}, whose scale is constant 1.0"
                )
            self.estimator.set_scale_provider(lambda: self.scaler.scale)
        self.warmup_steps = warmup_steps
        self.confidence_smoothing = float(np.clip(confidence_smoothing, 0.0, 1.0))
        self._confidence_ema = EMA(decay=confidence_decay)

    def observe(self, q: np.ndarray) -> None:
        self.scaler.observe(q)

    def knobs_for_confidence(self, c: float) -> tuple[float, float, int]:
        c = float(np.clip(c, 0.0, 1.0))
        eps = self.eps_uncertain + c * (self.eps_confident - self.eps_uncertain)
        # Geometric interpolation for the temperature: it spans orders of
        # magnitude, so linear interpolation would spend nearly the whole
        # confidence range in the neighbourhood of the larger endpoint.
        tau = math.exp(
            math.log(self.tau_uncertain)
            + c * (math.log(self.tau_confident) - math.log(self.tau_uncertain))
        )
        k = int(round(self.k_uncertain + c * (self.k_confident - self.k_uncertain)))
        k = int(np.clip(k, 1, self.num_actions))
        return eps, tau, k

    def action_probs(
        self, q: np.ndarray, step: int, uncertainty: float | None = None
    ) -> np.ndarray:
        raw_confidence = 0.0 if uncertainty is None else float(uncertainty)
        if step < self.warmup_steps:
            # Stay epsilon-greedy until the uncertainty reference is meaningful.
            raw_confidence = 0.0
        self._confidence_ema.update(raw_confidence)
        smoothed = self._confidence_ema.value
        c = (
            self.confidence_smoothing * raw_confidence
            + (1.0 - self.confidence_smoothing) * smoothed
        )

        eps, tau, k = self.knobs_for_confidence(c)
        q_tilde = self.scaler(q)
        support = top_k_support(q_tilde, k)
        # Build explicitly rather than splatting the estimator's diagnostics:
        # it also reports a `confidence` key (its own, pre-warmup-and-smoothing
        # value) and the two must not silently overwrite each other.
        info = dict(self.estimator.diagnostics())
        info["signal_confidence"] = info.pop("confidence", 0.0)
        info.update(
            eps=eps,
            temperature=tau,
            top_k=int(support.sum()),
            q_scale=self.scaler.scale,
            confidence=c,
            confidence_mean=smoothed,
        )
        self._last.update(info)
        return compose_probs(q_tilde, eps, tau, support)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scaler": self.scaler.state_dict(),
            "estimator": self.estimator.state_dict(),
            "confidence_ema": self._confidence_ema.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.scaler.load_state_dict(state["scaler"])
        self.estimator.load_state_dict(state["estimator"])
        self._confidence_ema.load_state_dict(state["confidence_ema"])
