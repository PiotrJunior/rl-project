"""Schedules must hit their endpoints exactly.

An anneal that stops at 0.98 instead of 1.0 would mean the 'fully Boltzmann'
arm never actually gets there, quietly turning every endpoint claim in the
report into an approximation.
"""

import numpy as np
import pytest

from e2b.utils.schedules import Constant, Exponential, Linear, LogLinear, make_schedule


def test_constant_from_bare_number():
    s = make_schedule(0.3)
    assert isinstance(s, Constant)
    assert s(0) == 0.3 and s(10**9) == 0.3


@pytest.mark.parametrize("kind", ["linear", "log_linear", "exponential"])
def test_endpoints_are_exact(kind):
    start, end, duration = (1.0, 0.01, 1000)
    s = make_schedule({"type": kind, "start": start, "end": end, "duration": duration})
    assert s(0) == pytest.approx(start, rel=1e-9)
    assert s(duration) == pytest.approx(end, rel=1e-9)
    assert s(duration * 10) == pytest.approx(end, rel=1e-9)


@pytest.mark.parametrize("kind", ["linear", "log_linear"])
def test_monotone_between_endpoints(kind):
    s = make_schedule({"type": kind, "start": 1.0, "end": 0.01, "duration": 1000})
    values = [s(t) for t in range(0, 1001, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_delay_holds_at_start():
    """Annealing towards Boltzmann should not begin while Q is still noise."""
    s = Linear(start=1.0, end=0.0, duration=100, delay=50)
    assert s(0) == 1.0
    assert s(50) == 1.0
    assert s(100) == pytest.approx(0.5)
    assert s(150) == pytest.approx(0.0)


def test_log_linear_is_geometric():
    s = LogLinear(start=1e-4, end=1.0, duration=100)
    # Halfway in log space is the geometric mean, not the arithmetic one.
    assert s(50) == pytest.approx(np.sqrt(1e-4 * 1.0))
    assert s(50) < 0.5 * (1e-4 + 1.0)


def test_log_linear_rejects_non_positive_endpoints():
    with pytest.raises(ValueError):
        LogLinear(start=0.0, end=1.0, duration=10)


def test_exponential_decays_towards_end():
    s = Exponential(start=1.0, end=0.0, duration=1000)
    assert s(0) == pytest.approx(1.0)
    assert 0.0 < s(500) < 1.0
    assert s(1000) == pytest.approx(0.0)


def test_zero_duration_is_immediately_at_end():
    s = Linear(start=1.0, end=0.0, duration=0)
    assert s(0) == 0.0


def test_unknown_schedule_type_raises():
    with pytest.raises(KeyError):
        make_schedule({"type": "quadratic", "start": 1, "end": 0, "duration": 10})
