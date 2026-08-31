"""Measures of how much to trust the current Q-estimate.

The project's extension replaces a *time-based* handover schedule with a
*measurement-based* one: switch towards Boltzmann where the Q-function is
trustworthy, and stay with epsilon-greedy where it is not.  That requires a
number, and the number has to be scale-free -- the raw magnitude of a
disagreement or a TD error means nothing on its own, since both shrink as
training converges and both differ by orders of magnitude between environments.

Each estimator therefore returns a raw signal *and* normalises it against a
running high-quantile of its own history, producing a confidence in [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from e2b.utils.running_stats import EMA, RunningQuantile


class UncertaintyEstimator(ABC):
    """Turns a raw uncertainty signal into a confidence in [0, 1]."""

    name = "base"

    def __init__(
        self,
        quantile: float = 0.9,
        quantile_lr: float = 0.01,
        reference_freeze_step: int = 0,
    ) -> None:
        self._reference = RunningQuantile(q=quantile, lr=quantile_lr)
        # 0 = never freeze (the default, and what the section 8.2 runs used).
        # A positive value stops updating the reference at that step, turning it
        # into a fixed calibration constant measured during early training.
        # Without this the reference is estimated from the same stream it
        # normalises, so it tracks the signal and `1 - u/ref` is pinned to a
        # constant however the signal moves -- see set_scale_provider.
        self.reference_freeze_step = int(reference_freeze_step)
        self._last_raw: float = 0.0
        self._last_confidence: float = 0.0
        self._scale_provider: Callable[[], float] | None = None

    def set_scale_provider(self, provider: Callable[[], float] | None) -> None:
        """Divide the raw signal by an external Q-scale before normalising.

        Both signals here are measured in **raw Q units**: ensemble
        disagreement is a standard deviation of Q-values, and the TD error is a
        difference of them. Q magnitudes grow substantially during training
        (the whole point of :mod:`e2b.policies.scaling`), so an *absolute*
        uncertainty can grow while the agent's *relative* uncertainty falls.
        Normalising against a running quantile of the signal's own history does
        not fix that -- the reference grows too, the ratio stays flat, and the
        confidence gate never moves.

        Supplying the same running Q-scale the policy uses for its temperature
        makes the signal dimensionless, so a shrinking uncertainty *relative to
        the size of the values* is what drives the handover. Off by default:
        the runs reported in the report's section 8.2 did not have it.
        """
        self._scale_provider = provider

    def _scale(self) -> float:
        if self._scale_provider is None:
            return 1.0
        return max(float(self._scale_provider()), 1e-8)

    @abstractmethod
    def raw(self, **kwargs: Any) -> float:
        """The unnormalised uncertainty signal (larger = less trustworthy)."""

    def confidence(self, **kwargs: Any) -> float:
        """Scale-free confidence: 1 when the signal is small relative to its own
        recent high-water mark, 0 when it is at or above it.

        Normalising against a *running* reference rather than a fixed constant
        is essential: uncertainty falls by orders of magnitude over training, so
        any fixed threshold would trip once and never move again, collapsing the
        adaptive scheme into a step function.
        """
        u = self.raw(**kwargs)
        self._last_raw = u
        reference = self._reference.estimate
        if not self._reference.initialised or reference <= 0.0:
            c = 0.0
        else:
            c = float(np.clip(1.0 - u / reference, 0.0, 1.0))
        self._last_confidence = c
        return c

    def update_reference(self, value: float | None = None,
                         step: int | None = None) -> None:
        """Fold a signal value into the running reference.

        Kept separate from :meth:`confidence` so the reference can be updated on
        the *training* path (many samples per step, low variance) while
        confidence is queried on the *acting* path.

        Ignored once ``step`` reaches ``reference_freeze_step``.
        """
        if self.reference_freeze_step > 0 and step is not None:
            if step >= self.reference_freeze_step:
                return
        self._reference.update(self._last_raw if value is None else value)

    def diagnostics(self) -> dict[str, float]:
        return {
            "uncertainty_raw": self._last_raw,
            "uncertainty_reference": self._reference.estimate,
            "confidence": self._last_confidence,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"reference": self._reference.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._reference.load_state_dict(state["reference"])


class EnsembleDisagreement(UncertaintyEstimator):
    """Epistemic uncertainty from spread across bootstrapped Q-heads.

    ``u(s) = mean_a std_k Q_k(s, a)``

    This is the flagship signal because it is genuinely **per-state**: the
    policy can be near-greedy-Boltzmann in a region it has visited thousands of
    times while remaining epsilon-greedy in a region it has barely seen. A
    global, time-based schedule cannot express that at all.

    It measures disagreement between heads trained on different bootstrap
    samples, so it reflects *what the data has not pinned down* rather than the
    intrinsic randomness of the return.
    """

    name = "ensemble"

    def raw(self, q_heads: np.ndarray | None = None, **kwargs: Any) -> float:
        if q_heads is None:
            raise ValueError("EnsembleDisagreement requires `q_heads`")
        q_heads = np.asarray(q_heads, dtype=np.float64)
        if q_heads.ndim != 2:
            raise ValueError(f"expected (num_heads, num_actions), got {q_heads.shape}")
        if q_heads.shape[0] < 2:
            return 0.0
        return float(q_heads.std(axis=0).mean()) / self._scale()


class TdErrorUncertainty(UncertaintyEstimator):
    """Global uncertainty proxy: an EMA of the absolute TD error.

    Free -- prioritized replay already computes per-sample TD errors on every
    training step, so this costs one EMA update.

    Its limitation is the point of comparison with the ensemble signal: it is a
    single scalar for the whole state space, so it can only reproduce a
    time-varying global schedule (albeit one driven by learning progress rather
    than by step count). Whether the per-state resolution of the ensemble is
    worth its extra compute is exactly what the two arms measure.
    """

    name = "td_error"

    def __init__(
        self,
        quantile: float = 0.9,
        quantile_lr: float = 0.01,
        reference_freeze_step: int = 0,
        decay: float = 0.99,
    ) -> None:
        super().__init__(quantile, quantile_lr, reference_freeze_step)
        self._ema = EMA(decay=decay)

    def observe_td_errors(self, td_errors: np.ndarray, step: int | None = None) -> float:
        """Fold a training batch's TD errors into the running estimate."""
        value = float(np.abs(np.asarray(td_errors, dtype=np.float64)).mean()) / self._scale()
        self._ema.update(value)
        self._last_raw = self._ema.value
        self.update_reference(self._ema.value, step=step)
        return self._ema.value

    def raw(self, **kwargs: Any) -> float:
        return self._ema.value if self._ema.initialised else 0.0

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["ema"] = self._ema.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._ema.load_state_dict(state["ema"])


_REGISTRY = {
    "ensemble": EnsembleDisagreement,
    "td_error": TdErrorUncertainty,
}


def build_estimator(name: str, **kwargs: Any) -> UncertaintyEstimator:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown uncertainty signal {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
