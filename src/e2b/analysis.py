"""Aggregation of results across seeds.

Deep-RL results are frequently reported as a mean over 3-5 seeds with no
interval, which is not enough to distinguish a real effect from seed noise --
the point Agarwal et al. (2021), "Deep RL at the Edge of the Statistical
Precipice", make at length. This module implements the two tools from that paper
that matter at small seed counts, without taking on ``rliable`` as a dependency:

* **IQM** (interquartile mean): the mean of the middle 50% of runs. Far less
  sensitive than the mean to the single catastrophic-failure seed that DQN
  produces regularly, and much less noisy than the median.
* **Stratified bootstrap CIs**: resample seeds *within* each arm, so the
  interval reflects the seed variance actually present.

With 3 seeds the intervals are wide, and they are supposed to be: the honest
conclusion from 3 seeds is usually "inconclusive", and the report says so rather
than reading a ranking out of noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return 0.5 * (self.high - self.low)

    def __str__(self) -> str:
        return f"{self.point:.1f} [{self.low:.1f}, {self.high:.1f}] (n={self.n})"


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of the values between the 25th and 75th centiles.

    Falls back to the plain mean for fewer than 4 values, where the trimmed
    range is not meaningful.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(arr.mean())
    lo, hi = np.percentile(arr, [25, 75])
    middle = arr[(arr >= lo) & (arr <= hi)]
    return float(middle.mean() if middle.size else arr.mean())


def bootstrap_interval(
    values: np.ndarray,
    statistic=iqm,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for ``statistic`` over seeds.

    Resampling is over *runs* (seeds), which is the unit of independent
    replication here -- resampling over evaluation points instead would treat
    correlated measurements from one run as independent and produce intervals
    that are far too narrow.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        return Interval(float(arr[0]), float(arr[0]), float(arr[0]), 1)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    stats = np.array([statistic(arr[idx]) for idx in draws])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return Interval(float(statistic(arr)), float(low), float(high), n)


def curve_interval(
    curves: np.ndarray, confidence: float = 0.95, resamples: int = 2_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step IQM and CI band for a stack of learning curves.

    ``curves`` is ``(num_seeds, num_eval_points)``. Seeds are resampled *once
    per bootstrap replicate and applied to every step*, not independently per
    step: a seed is a whole trajectory, and resampling within steps would
    manufacture curves that no run ever produced and understate the band.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError(f"expected (seeds, steps), got {curves.shape}")
    n_seeds, n_steps = curves.shape
    point = np.array([iqm(curves[:, t]) for t in range(n_steps)])
    if n_seeds < 2:
        return point, point.copy(), point.copy()

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seeds, size=(resamples, n_seeds))
    stats = np.empty((resamples, n_steps))
    for r, idx in enumerate(draws):
        sample = curves[idx]
        stats[r] = [iqm(sample[:, t]) for t in range(n_steps)]
    alpha = (1.0 - confidence) / 2.0
    low = np.percentile(stats, 100 * alpha, axis=0)
    high = np.percentile(stats, 100 * (1 - alpha), axis=0)
    return point, low, high


def steps_to_threshold(
    steps: np.ndarray, returns: np.ndarray, threshold: float
) -> float:
    """First evaluation step at which the curve reaches ``threshold``.

    The sample-efficiency metric. Returns ``inf`` for a run that never gets
    there, which is deliberate -- substituting the final step would make a run
    that failed look merely slow, and the two deserve to be distinguished.
    """
    reached = np.flatnonzero(np.asarray(returns) >= threshold)
    if reached.size == 0:
        return float("inf")
    return float(np.asarray(steps)[reached[0]])


def probability_of_improvement(
    a: np.ndarray, b: np.ndarray, resamples: int = 10_000, seed: int = 0
) -> Interval:
    """P(a random run of A beats a random run of B), with a bootstrap CI.

    More informative than comparing means at small seed counts: it asks the
    question a practitioner actually has ("if I run this once, will it be
    better?") and is robust to one outlying seed. 0.5 means indistinguishable.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    def stat(x: np.ndarray, y: np.ndarray) -> float:
        # Ties count as half, the usual Mann-Whitney convention.
        comparison = np.sign(x[:, None] - y[None, :])
        return float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())

    rng = np.random.default_rng(seed)
    point = stat(a, b)
    stats = np.array([
        stat(a[rng.integers(0, a.size, a.size)], b[rng.integers(0, b.size, b.size)])
        for _ in range(resamples)
    ])
    low, high = np.percentile(stats, [2.5, 97.5])
    return Interval(point, float(low), float(high), min(a.size, b.size))
