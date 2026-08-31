"""Scalar schedules over the environment-step counter.

Every exploration knob in this project (epsilon, temperature, top-k support
size, mixing coefficient) is driven by one of these.  Keeping them in one place
means a variant is fully described by its schedule config, and the limiting-case
tests in ``tests/test_schedules.py`` only have to be written once.

All schedules are defined on an absolute step ``t`` and are pure functions of it,
so a run can be checkpointed and resumed without carrying schedule state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class Schedule:
    """Base class: a pure function from environment step to a scalar."""

    def __call__(self, t: int) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def progress(self, t: int) -> float:
        """Fraction of the schedule completed at step ``t``, clipped to [0, 1].

        Used by the composite exploration policies to keep several knobs in
        lockstep without duplicating the anneal window.
        """
        raise NotImplementedError


@dataclass
class Constant(Schedule):
    value: float

    def __call__(self, t: int) -> float:
        return self.value

    def progress(self, t: int) -> float:
        return 1.0


@dataclass
class Linear(Schedule):
    """Linear interpolation from ``start`` to ``end`` over ``duration`` steps.

    Held at ``start`` for the first ``delay`` steps.  ``delay`` exists because
    it is usually pointless to start annealing towards Boltzmann while the
    replay buffer is still warming up and Q is pure noise.
    """

    start: float
    end: float
    duration: int
    delay: int = 0

    def progress(self, t: int) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (t - self.delay) / self.duration))

    def __call__(self, t: int) -> float:
        p = self.progress(t)
        return self.start + p * (self.end - self.start)


@dataclass
class Exponential(Schedule):
    """Geometric decay from ``start`` towards ``end``.

    Parameterised by the duration over which the *remaining* gap shrinks to
    ``1 - decay_fraction`` of its initial size, so it is directly comparable to
    a ``Linear`` schedule with the same ``duration``.

    Temperature annealing is much more natural on a log scale than a linear one
    (going 1.0 -> 0.1 -> 0.01 are equal-sized changes in behaviour, but wildly
    unequal in linear space), which is why this exists alongside ``Linear``.
    """

    start: float
    end: float
    duration: int
    delay: int = 0
    decay_fraction: float = 0.99

    def progress(self, t: int) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (t - self.delay) / self.duration))

    def __call__(self, t: int) -> float:
        p = self.progress(t)
        if p >= 1.0:
            return self.end
        # gap shrinks geometrically so that at p == 1 it has shrunk to
        # (1 - decay_fraction) of the initial gap.
        remaining = (1.0 - self.decay_fraction) ** p
        return self.end + (self.start - self.end) * remaining


@dataclass
class LogLinear(Schedule):
    """Linear in log-space: geometric interpolation from ``start`` to ``end``.

    The natural schedule for a temperature.  Both endpoints must be > 0.
    Unlike :class:`Exponential` this hits ``end`` exactly at ``p == 1``.
    """

    start: float
    end: float
    duration: int
    delay: int = 0

    def __post_init__(self) -> None:
        if self.start <= 0.0 or self.end <= 0.0:
            raise ValueError(
                f"LogLinear requires strictly positive endpoints, got "
                f"start={self.start}, end={self.end}"
            )

    def progress(self, t: int) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, max(0.0, (t - self.delay) / self.duration))

    def __call__(self, t: int) -> float:
        p = self.progress(t)
        return math.exp(
            math.log(self.start) + p * (math.log(self.end) - math.log(self.start))
        )


_REGISTRY: dict[str, type[Schedule]] = {
    "constant": Constant,
    "linear": Linear,
    "exponential": Exponential,
    "log_linear": LogLinear,
}


def make_schedule(spec: Mapping[str, Any] | float | int | Schedule) -> Schedule:
    """Build a schedule from a config fragment.

    A bare number is shorthand for a constant schedule, which keeps configs
    readable when a knob is not being annealed::

        temperature: 0.5                       # constant
        temperature: {type: log_linear, start: 1.0e-3, end: 0.5, duration: 100000}
    """
    if isinstance(spec, Schedule):
        return spec
    if isinstance(spec, (int, float)):
        return Constant(float(spec))
    if not isinstance(spec, Mapping):
        raise TypeError(f"cannot build a schedule from {spec!r}")

    kwargs = dict(spec)
    kind = kwargs.pop("type", "constant")
    if kind not in _REGISTRY:
        raise KeyError(
            f"unknown schedule type {kind!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind](**kwargs)
