"""Aggregation statistics.

These decide what the report is allowed to claim, so they need to be right.
"""

import numpy as np
import pytest

from e2b.analysis import (
    bootstrap_interval,
    curve_interval,
    iqm,
    probability_of_improvement,
    steps_to_threshold,
)


def test_iqm_resists_a_catastrophic_seed():
    """The reason IQM is used instead of the mean.

    DQN produces total-failure seeds regularly; one of them moves a 5-seed mean
    by hundreds of points, which is enough to invert a ranking.
    """
    good = np.array([100.0, 105.0, 110.0, 115.0, -500.0])
    assert iqm(good) == pytest.approx(105.0)
    assert good.mean() < 0.0


def test_iqm_falls_back_to_mean_for_small_samples():
    v = np.array([1.0, 3.0, 5.0])
    assert iqm(v) == pytest.approx(3.0)


def test_iqm_ignores_non_finite_values():
    assert iqm(np.array([1.0, np.nan, 3.0, np.inf])) == pytest.approx(2.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 10.0, size=8)
    ci = bootstrap_interval(values, resamples=2000)
    assert ci.low <= ci.point <= ci.high
    assert ci.n == 8


def test_bootstrap_interval_narrows_with_more_seeds():
    """The property that makes the interval informative rather than decorative."""
    rng = np.random.default_rng(0)
    narrow = bootstrap_interval(rng.normal(0, 1, 50), resamples=2000)
    wide = bootstrap_interval(rng.normal(0, 1, 4), resamples=2000)
    assert narrow.half_width < wide.half_width


def test_bootstrap_interval_handles_degenerate_inputs():
    assert bootstrap_interval(np.array([])).n == 0
    single = bootstrap_interval(np.array([5.0]))
    assert single.point == 5.0 and single.low == 5.0 and single.high == 5.0


def test_curve_interval_resamples_whole_seeds_not_points():
    """A seed is a trajectory. Resampling within steps would manufacture curves
    no run produced and understate the band."""
    # Two seeds, perfectly separated and each internally constant.
    curves = np.array([[0.0] * 5, [10.0] * 5])
    point, low, high = curve_interval(curves, resamples=500, seed=0)
    # Every bootstrap replicate must be a constant curve (0, 5, or 10) -- never
    # a mixture that varies across steps within one replicate.
    assert np.allclose(low, low[0])
    assert np.allclose(high, high[0])
    assert np.allclose(point, 5.0)


def test_curve_interval_band_contains_point_estimate():
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(6, 12)) + np.arange(12)
    point, low, high = curve_interval(curves, resamples=300, seed=0)
    assert np.all(low <= point + 1e-9)
    assert np.all(point <= high + 1e-9)


def test_curve_interval_rejects_wrong_shape():
    with pytest.raises(ValueError):
        curve_interval(np.zeros(5))


def test_steps_to_threshold_reports_first_crossing():
    steps = np.array([10, 20, 30, 40])
    assert steps_to_threshold(steps, np.array([0.0, 1.0, 5.0, 9.0]), 5.0) == 30.0


def test_steps_to_threshold_is_infinite_when_never_reached():
    """A run that failed must not be reported as merely slow."""
    assert steps_to_threshold(np.array([10, 20]), np.array([0.0, 1.0]), 99.0) == float("inf")


def test_probability_of_improvement_is_half_for_identical_samples():
    v = np.array([1.0, 2.0, 3.0])
    assert probability_of_improvement(v, v, resamples=500).point == pytest.approx(0.5)


def test_probability_of_improvement_saturates_for_separated_samples():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert probability_of_improvement(a, b, resamples=500).point == pytest.approx(1.0)
    assert probability_of_improvement(b, a, resamples=500).point == pytest.approx(0.0)
