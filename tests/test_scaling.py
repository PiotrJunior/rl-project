"""Q-scale normalisation.

The claim being tested is the project's main implementation finding: `running`
normalisation makes a temperature portable across environments and across
training time, while `per_state` normalisation destroys the "all actions are
equally good" signal and `none` leaves the temperature at the mercy of Q drift.
"""

import numpy as np
import pytest

from e2b.policies.core import masked_softmax
from e2b.policies.scaling import QScaler


def test_none_mode_is_shift_invariant_only():
    scaler = QScaler("none")
    q = np.array([1.0, 2.0, 3.0])
    out = scaler(q)
    # Centring is a no-op for softmax, so relative gaps must be preserved.
    np.testing.assert_allclose(out - out[0], q - q[0])


def test_per_state_forces_unit_spread_even_on_a_flat_q_row():
    """The failure mode: a genuinely undecided state is rescaled into confidence."""
    scaler = QScaler("per_state")
    flat = np.array([1.0000, 1.0001, 0.9999, 1.0000])
    out = scaler(flat)
    assert out.std() == pytest.approx(1.0, rel=1e-3)
    # A softmax at a moderate temperature is now strongly peaked on noise.
    probs = masked_softmax(out, np.ones(4, dtype=bool), 0.3)
    assert probs.max() > 0.5


def test_running_mode_keeps_a_flat_q_row_near_uniform():
    """The same flat row under `running` stays close to uniform, as it should."""
    scaler = QScaler("running", decay=0.9)
    rng = np.random.default_rng(0)
    # Establish a scale from states with a real spread.
    for _ in range(500):
        scaler.observe(rng.normal(size=4) * 10.0)
    flat = np.array([1.0000, 1.0001, 0.9999, 1.0000])
    probs = masked_softmax(scaler(flat), np.ones(4, dtype=bool), 0.3)
    np.testing.assert_allclose(probs, np.full(4, 0.25), atol=1e-3)


def test_running_mode_makes_temperature_invariant_to_q_magnitude():
    """The property that makes one temperature work on CartPole and LunarLander.

    Two Q-rows with the same *shape* but magnitudes differing by 1000x must
    produce the same action distribution.
    """
    shape = np.array([0.0, 1.0, 0.5, -1.0])

    def distribution(scale):
        # A fresh RNG per call, so the two runs see the *same* state sequence
        # differing only by the scale factor. The invariance is then exact
        # rather than statistical: std(scale * x) == scale * std(x), so the
        # running scale is exactly `scale` times larger and cancels.
        rng = np.random.default_rng(0)
        scaler = QScaler("running", decay=0.99)
        for _ in range(2000):
            scaler.observe(rng.normal(size=4) * scale)
        return masked_softmax(scaler(shape * scale), np.ones(4, dtype=bool), 0.3)

    np.testing.assert_allclose(distribution(1.0), distribution(1000.0), rtol=1e-9)


def test_none_mode_temperature_is_not_invariant_to_q_magnitude():
    """Contrast case: without normalisation the same temperature behaves
    completely differently at different Q scales."""
    shape = np.array([0.0, 1.0, 0.5, -1.0])
    scaler = QScaler("none")
    small = masked_softmax(scaler(shape * 1.0), np.ones(4, dtype=bool), 0.3)
    large = masked_softmax(scaler(shape * 1000.0), np.ones(4, dtype=bool), 0.3)
    assert small.max() < 0.9        # meaningfully stochastic
    assert large.max() > 0.999      # collapsed to greedy


def test_scale_is_positive_and_survives_a_roundtrip():
    scaler = QScaler("running")
    for _ in range(100):
        scaler.observe(np.array([1.0, 2.0, 3.0, 4.0]))
    state = scaler.state_dict()
    restored = QScaler("running")
    restored.load_state_dict(state)
    assert restored.scale == pytest.approx(scaler.scale)
    assert scaler.scale > 0.0


def test_unknown_mode_rejected():
    with pytest.raises(KeyError):
        QScaler("magic")
