"""Uncertainty estimators and the gated policy's response to them."""

import numpy as np
import pytest

from e2b.policies import build_policy
from e2b.uncertainty import EnsembleDisagreement, TdErrorUncertainty, build_estimator


def test_ensemble_disagreement_is_zero_for_identical_heads():
    est = EnsembleDisagreement()
    agreed = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    assert est.raw(q_heads=agreed) == pytest.approx(0.0)


def test_ensemble_disagreement_grows_with_spread():
    est = EnsembleDisagreement()
    rng = np.random.default_rng(0)
    low = est.raw(q_heads=rng.normal(size=(5, 3)) * 0.1)
    high = est.raw(q_heads=rng.normal(size=(5, 3)) * 10.0)
    assert high > low


def test_single_head_reports_no_disagreement():
    """A 1-head 'ensemble' cannot measure anything; it must say so rather than
    emit a spurious value that would gate the policy."""
    assert EnsembleDisagreement().raw(q_heads=np.zeros((1, 3))) == 0.0


def test_confidence_is_scale_free():
    """The same *relative* uncertainty must give the same confidence whether the
    raw signal is order 0.01 or order 1000."""

    def final_confidence(scale):
        est = EnsembleDisagreement()
        rng = np.random.default_rng(0)
        for _ in range(2000):
            est.confidence(q_heads=rng.normal(size=(5, 3)) * scale)
            est.update_reference()
        # Now query with a low-disagreement state at the same scale.
        return est.confidence(q_heads=np.tile(rng.normal(size=3) * scale, (5, 1)))

    assert final_confidence(0.01) == pytest.approx(final_confidence(1000.0), abs=0.05)


def test_confidence_is_bounded():
    est = EnsembleDisagreement()
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = est.confidence(q_heads=rng.normal(size=(5, 3)))
        est.update_reference()
        assert 0.0 <= c <= 1.0


def test_confidence_rises_as_heads_converge():
    """The behaviour the extension depends on: as the ensemble agrees more, the
    policy should be told it can trust Q."""
    est = EnsembleDisagreement()
    rng = np.random.default_rng(0)
    for _ in range(1000):          # establish a reference at high disagreement
        est.confidence(q_heads=rng.normal(size=(5, 3)) * 5.0)
        est.update_reference()
    confident = est.confidence(q_heads=np.tile(np.array([1.0, 2.0, 3.0]), (5, 1)))
    uncertain = est.confidence(q_heads=rng.normal(size=(5, 3)) * 50.0)
    assert confident > 0.9
    assert uncertain < 0.1


def test_td_error_uncertainty_tracks_the_error_magnitude():
    est = TdErrorUncertainty(decay=0.5)
    for _ in range(50):
        est.observe_td_errors(np.array([10.0]))
    high = est.raw()
    for _ in range(50):
        est.observe_td_errors(np.array([0.01]))
    assert est.raw() < high


def test_gated_policy_is_epsilon_greedy_at_zero_confidence():
    """Confidence 0 must land exactly on the epsilon-greedy endpoint."""
    policy = build_policy(
        {"name": "uncertainty_gated", "eps_uncertain": 0.3, "eps_confident": 0.01,
         "k_uncertain": 1, "k_confident": "num_actions", "warmup_steps": 0,
         "q_scaling": "none"}, 4)
    eps, _, k = policy.knobs_for_confidence(0.0)
    assert eps == pytest.approx(0.3)
    assert k == 1


def test_gated_policy_is_boltzmann_at_full_confidence():
    policy = build_policy(
        {"name": "uncertainty_gated", "eps_uncertain": 0.3, "eps_confident": 0.01,
         "tau_uncertain": 1.0, "tau_confident": 0.2,
         "k_uncertain": 1, "k_confident": "num_actions", "warmup_steps": 0,
         "q_scaling": "none"}, 4)
    eps, tau, k = policy.knobs_for_confidence(1.0)
    assert eps == pytest.approx(0.01)
    assert tau == pytest.approx(0.2)
    assert k == 4


def test_gated_policy_stays_epsilon_greedy_during_warmup():
    """Before the reference quantile is meaningful, a spurious high confidence
    must not be allowed to start the run in near-greedy Boltzmann on a randomly
    initialised network."""
    policy = build_policy(
        {"name": "uncertainty_gated", "warmup_steps": 100, "q_scaling": "none",
         "eps_uncertain": 1.0}, 4)
    probs = policy.action_probs(np.array([1.0, 5.0, 2.0, 0.0]), step=0, uncertainty=1.0)
    np.testing.assert_allclose(probs, np.full(4, 0.25), atol=1e-9)
    assert policy.diagnostics()["confidence"] == pytest.approx(0.0)


def test_gated_knobs_are_monotone_in_confidence():
    policy = build_policy({"name": "uncertainty_gated", "warmup_steps": 0}, 4)
    grid = [policy.knobs_for_confidence(c) for c in np.linspace(0, 1, 11)]
    eps = [g[0] for g in grid]
    ks = [g[2] for g in grid]
    assert all(a >= b for a, b in zip(eps, eps[1:]))    # epsilon falls
    assert all(a <= b for a, b in zip(ks, ks[1:]))      # support widens


def test_confidence_smoothing_zero_gives_a_global_signal():
    """The ablation that separates 'adaptive timing' from 'per-state resolution'."""
    policy = build_policy(
        {"name": "uncertainty_gated", "warmup_steps": 0, "confidence_smoothing": 0.0,
         "confidence_decay": 0.9, "q_scaling": "none"}, 4)
    q = np.array([1.0, 5.0, 2.0, 0.0])
    for _ in range(200):
        policy.action_probs(q, step=10, uncertainty=0.5)
    # A single outlying per-state value barely moves a fully-smoothed signal.
    policy.action_probs(q, step=10, uncertainty=1.0)
    assert policy.diagnostics()["confidence"] < 0.6


def test_unknown_signal_rejected():
    with pytest.raises(KeyError):
        build_estimator("telepathy")


# --- Q-scale normalisation of the uncertainty signal -------------------------
#
# The first iteration of the extension failed because both signals are measured
# in raw Q units while Q grows during training, so the signal grew too and the
# confidence gate never moved. These pin down the fix.


def test_raw_signal_is_unnormalised_by_default():
    """The section 8.2 runs had no scale provider; that behaviour must not change."""
    est = EnsembleDisagreement()
    q_heads = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert est.raw(q_heads=q_heads) == pytest.approx(float(q_heads.std(axis=0).mean()))


def test_scale_provider_makes_ensemble_disagreement_dimensionless():
    """Doubling every Q-value doubles the disagreement AND the scale, so the
    normalised signal is unchanged. This invariance is the whole point."""
    est = EnsembleDisagreement()
    scale = 1.0
    est.set_scale_provider(lambda: scale)
    q_heads = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 2.0]])

    base = est.raw(q_heads=q_heads)
    for factor in (0.01, 7.0, 1000.0):
        scale = factor
        assert est.raw(q_heads=q_heads * factor) == pytest.approx(base, rel=1e-12)


def test_scale_provider_normalises_the_td_error_signal_too():
    est = TdErrorUncertainty(decay=0.5)
    scale = 4.0
    est.set_scale_provider(lambda: scale)
    est.observe_td_errors(np.array([8.0, 8.0]))
    assert est.raw() == pytest.approx(2.0)


def test_gated_policy_wires_its_own_q_scale_into_the_estimator():
    policy = build_policy(
        {"name": "uncertainty_gated", "warmup_steps": 0,
         "normalise_uncertainty": True, "q_scaling": "running"}, 3)
    # Feed the scaler a stream of Q-rows so it has a running spread > 0.
    for _ in range(500):
        policy.observe(np.array([0.0, 10.0, 20.0]))
    scale = policy.scaler.scale
    assert scale > 1.0
    q_heads = np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    expected = float(q_heads.std(axis=0).mean()) / scale
    assert policy.estimator.raw(q_heads=q_heads) == pytest.approx(expected)


def test_normalise_uncertainty_rejects_a_constant_scale_mode():
    """per_state and none both return scale 1.0, so this would silently no-op."""
    for mode in ("none", "per_state"):
        with pytest.raises(ValueError, match="q_scaling='running'"):
            build_policy(
                {"name": "uncertainty_gated", "normalise_uncertainty": True,
                 "q_scaling": mode}, 3)


def test_frozen_reference_stops_tracking_the_signal():
    """The structural fix: a reference estimated from the stream it normalises
    follows that stream, so `1 - u/ref` cannot move. Freezing it breaks the
    feedback and lets a genuinely falling uncertainty raise confidence."""
    est = EnsembleDisagreement(reference_freeze_step=10)
    high = np.array([[0.0, 0.0], [4.0, 4.0]])      # large disagreement
    for step in range(10):
        est.confidence(q_heads=high)
        est.update_reference(step=step)
    frozen = est.diagnostics()["uncertainty_reference"]
    assert frozen > 0.0

    low = np.array([[0.0, 0.0], [0.1, 0.1]])       # uncertainty collapses
    for step in range(10, 400):
        c = est.confidence(q_heads=low)
        est.update_reference(step=step)
    assert est.diagnostics()["uncertainty_reference"] == pytest.approx(frozen)
    assert c > 0.9      # confidence rose, because the reference held still


def test_unfrozen_reference_is_the_default_and_does_track():
    """Contrast case: the section 8.2 behaviour, kept as the default."""
    est = EnsembleDisagreement()
    high = np.array([[0.0, 0.0], [4.0, 4.0]])
    for step in range(10):
        est.confidence(q_heads=high)
        est.update_reference(step=step)
    early = est.diagnostics()["uncertainty_reference"]

    low = np.array([[0.0, 0.0], [0.1, 0.1]])
    for step in range(10, 400):
        est.confidence(q_heads=low)
        est.update_reference(step=step)
    assert est.diagnostics()["uncertainty_reference"] < early
