"""Running scalar statistics used to make temperatures scale-free.

The single most important implementation detail in this project is that a
Boltzmann temperature is *not* comparable across environments or across time
unless the Q-values fed to the softmax are normalised.  Raw Q magnitudes differ
by orders of magnitude between CartPole (~10-100) and LunarLander (~-200-300),
and they grow substantially during training within a single run.  A temperature
that produces sensible exploration at 50k steps can be effectively greedy at
500k steps with no change to the schedule.

These estimators supply the normalising scale.  See
``e2b.policies.unified.QScaler`` for how they are applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EMA:
    """Exponential moving average with bias correction.

    Bias correction matters here: without it the first few thousand steps
    report a scale biased towards the (arbitrary) initial value, which is
    exactly the period during which the exploration policy is most sensitive.
    """

    decay: float = 0.999
    _value: float = 0.0
    _debias: float = 0.0

    def update(self, x: float) -> float:
        self._value = self.decay * self._value + (1.0 - self.decay) * float(x)
        self._debias = self.decay * self._debias + (1.0 - self.decay)
        return self.value

    @property
    def value(self) -> float:
        if self._debias <= 0.0:
            return 0.0
        return self._value / self._debias

    @property
    def initialised(self) -> bool:
        return self._debias > 0.0

    def state_dict(self) -> dict[str, float]:
        return {"decay": self.decay, "value": self._value, "debias": self._debias}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.decay = state["decay"]
        self._value = state["value"]
        self._debias = state["debias"]


@dataclass
class RunningMoments:
    """Welford-style running mean/variance over a stream of scalars."""

    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        if self.count < 2.0:
            return 0.0
        return self.m2 / (self.count - 1.0)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.var))

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.count = state["count"]
        self.mean = state["mean"]
        self.m2 = state["m2"]


@dataclass
class RunningQuantile:
    """Online estimate of a quantile via the Frugal-2U / stochastic-approximation update.

    Used by the uncertainty-gated policy to obtain a reference scale ``u_ref``
    for the raw uncertainty signal without storing a history buffer.  We only
    need a rough high-quantile to normalise against, so a cheap stochastic
    estimator is entirely adequate and keeps the acting path allocation-free.

    The step size adapts to the observed spread of the signal, otherwise a fixed
    step size is meaningless for a quantity whose scale we do not know a priori.
    """

    q: float = 0.9
    lr: float = 0.01
    estimate: float = 0.0
    _moments: RunningMoments = field(default_factory=RunningMoments)

    def update(self, x: float) -> float:
        x = float(x)
        self._moments.update(x)
        if self._moments.count < 2.0:
            self.estimate = x
            return self.estimate
        # Scale the step by the observed spread so `lr` is dimensionless.
        step = self.lr * max(self._moments.std, 1e-8)
        if x > self.estimate:
            self.estimate += step * self.q
        elif x < self.estimate:
            self.estimate -= step * (1.0 - self.q)
        return self.estimate

    @property
    def initialised(self) -> bool:
        return self._moments.count > 0.0

    def state_dict(self) -> dict[str, object]:
        return {
            "q": self.q,
            "lr": self.lr,
            "estimate": self.estimate,
            "moments": self._moments.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.q = float(state["q"])  # type: ignore[arg-type]
        self.lr = float(state["lr"])  # type: ignore[arg-type]
        self.estimate = float(state["estimate"])  # type: ignore[arg-type]
        self._moments.load_state_dict(state["moments"])  # type: ignore[arg-type]
