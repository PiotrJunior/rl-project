"""Limiting-case equivalences for the unified exploration family.

These are the load-bearing tests of the project. The whole argument rests on
epsilon-greedy and Boltzmann being two points in one parameter space, so that
"annealing between them" is well defined. If the endpoints are only
*approximately* the classic strategies, then every reported comparison is
against something that is not quite the baseline it claims to be.

Because policies expose explicit probability vectors, these are exact array
comparisons rather than statistical tests on sampled actions.
"""

from __future__ import annotations

import numpy as np
import pytest

from e2b.policies import build_policy, masked_softmax, top_k_support, top_p_support
from e2b.policies.core import compose_probs

Q = np.array([1.0, 3.0, 2.5, -1.0])
NA = 4


def eps_greedy_reference(q: np.ndarray, eps: float) -> np.ndarray:
    """Textbook epsilon-greedy, written independently of the implementation."""
    n = q.shape[0]
    probs = np.full(n, eps / n)
    best = np.flatnonzero(q >= q.max())
    probs[best] += (1.0 - eps) / best.size
    return probs


def boltzmann_reference(q: np.ndarray, tau: float) -> np.ndarray:
    z = q / tau
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.5, 1.0])
@pytest.mark.parametrize("tau", [1e-6, 0.1, 1.0, 100.0])
def test_top_k_one_is_exactly_epsilon_greedy(eps, tau):
    """k = 1 collapses the softmax support to the argmax set, for ANY temperature."""
    policy = build_policy(
        {"name": "unified", "epsilon": eps, "temperature": tau,
         "top_k": 1, "q_scaling": "none"}, NA)
    np.testing.assert_allclose(
        policy.action_probs(Q, 0), eps_greedy_reference(Q, eps), atol=1e-12
    )


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.5])
def test_zero_temperature_limit_is_epsilon_greedy(eps):
    """tau -> 0 over the FULL action set also reduces to epsilon-greedy.

    This is the other endpoint of the temperature path, and it must agree with
    the k = 1 endpoint exactly -- otherwise the two 'paths' in the study start
    from different places.
    """
    policy = build_policy(
        {"name": "unified", "epsilon": eps, "temperature": 1e-12,
         "top_k": "num_actions", "q_scaling": "none"}, NA)
    np.testing.assert_allclose(
        policy.action_probs(Q, 0), eps_greedy_reference(Q, eps), atol=1e-12
    )


@pytest.mark.parametrize("tau", [0.05, 0.5, 1.0, 10.0])
def test_full_support_zero_epsilon_is_pure_boltzmann(tau):
    policy = build_policy(
        {"name": "unified", "epsilon": 0.0, "temperature": tau,
         "top_k": "num_actions", "q_scaling": "none"}, NA)
    np.testing.assert_allclose(
        policy.action_probs(Q, 0), boltzmann_reference(Q, tau), atol=1e-12
    )


def test_tie_breaking_is_uniform_over_argmax_set():
    """Both endpoints must spread mass uniformly over tied maxima.

    Ties are the norm at initialisation (a freshly initialised network produces
    near-identical Q for every action), so index-based tie-breaking would bias
    early exploration towards low-numbered actions in some arms but not others.
    """
    q = np.array([2.0, 2.0, 1.0, 0.0])
    expected = np.array([0.5, 0.5, 0.0, 0.0])
    by_k = build_policy(
        {"name": "unified", "epsilon": 0.0, "temperature": 3.0,
         "top_k": 1, "q_scaling": "none"}, NA).action_probs(q, 0)
    by_tau = build_policy(
        {"name": "unified", "epsilon": 0.0, "temperature": 1e-12,
         "top_k": "num_actions", "q_scaling": "none"}, NA).action_probs(q, 0)
    np.testing.assert_allclose(by_k, expected, atol=1e-12)
    np.testing.assert_allclose(by_tau, expected, atol=1e-12)


def test_mixture_endpoints_recover_each_component():
    """beta = 0 is exactly epsilon-greedy; beta = 1 is exactly Boltzmann."""
    first = {"epsilon": 0.1, "top_k": 1, "q_scaling": "none"}
    second = {"epsilon": 0.0, "temperature": 0.7, "q_scaling": "none",
              "top_k": "num_actions"}

    at_zero = build_policy(
        {"name": "mixture_anneal", "beta": 0.0, "first": first, "second": second}, NA)
    np.testing.assert_allclose(
        at_zero.action_probs(Q, 0), eps_greedy_reference(Q, 0.1), atol=1e-12)

    at_one = build_policy(
        {"name": "mixture_anneal", "beta": 1.0, "first": first, "second": second}, NA)
    np.testing.assert_allclose(
        at_one.action_probs(Q, 0), boltzmann_reference(Q, 0.7), atol=1e-12)


def test_mixture_is_linear_in_beta():
    """The mixture must interpolate distributions, not parameters.

    This is what distinguishes idea 1a from idea 1b, so it needs to actually
    hold rather than being approximated by coin flips.
    """
    first = {"epsilon": 0.1, "top_k": 1, "q_scaling": "none"}
    second = {"epsilon": 0.0, "temperature": 0.7, "q_scaling": "none",
              "top_k": "num_actions"}
    p0 = eps_greedy_reference(Q, 0.1)
    p1 = boltzmann_reference(Q, 0.7)
    for beta in (0.25, 0.5, 0.75):
        policy = build_policy(
            {"name": "mixture_anneal", "beta": beta, "first": first, "second": second}, NA)
        np.testing.assert_allclose(
            policy.action_probs(Q, 0), (1 - beta) * p0 + beta * p1, atol=1e-12)


def test_epsilon_one_is_uniform_regardless_of_other_knobs():
    for tau, k in [(1e-9, 1), (0.3, 2), (10.0, NA)]:
        policy = build_policy(
            {"name": "unified", "epsilon": 1.0, "temperature": tau,
             "top_k": k, "q_scaling": "none"}, NA)
        np.testing.assert_allclose(
            policy.action_probs(Q, 0), np.full(NA, 1 / NA), atol=1e-12)


# --------------------------------------------------------------- numerics

@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6, 1e9])
@pytest.mark.parametrize("tau", [1e-12, 1e-6, 1.0, 1e3])
def test_softmax_never_overflows_or_produces_nan(scale, tau):
    """The naive exp(q / tau) overflows for most of this grid.

    Both large |Q| and tiny tau occur routinely in a real run -- Q grows as the
    value function inflates, and tau is annealed down -- so this is a realistic
    grid, not a pathological one.
    """
    q = Q * scale
    probs = compose_probs(q, 0.0, tau, np.ones(NA, dtype=bool))
    assert np.all(np.isfinite(probs)), probs
    assert probs.min() >= 0.0
    assert abs(probs.sum() - 1.0) < 1e-12
    # At extreme scale/temperature ratios the limit must be the argmax, not NaN.
    if scale / tau > 1e6:
        assert probs.argmax() == q.argmax()


def test_masked_softmax_puts_zero_mass_outside_support():
    support = np.array([True, False, True, False])
    probs = masked_softmax(Q, support, 0.5)
    assert probs[1] == 0.0 and probs[3] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-12


def test_epsilon_floor_reaches_actions_excluded_from_top_k():
    """A non-zero epsilon must keep excluded actions reachable.

    Otherwise an action wrongly ranked low by a noisy Q is permanently
    unreachable, which is precisely the failure mode top-k exploration is
    supposed to avoid.
    """
    policy = build_policy(
        {"name": "unified", "epsilon": 0.1, "temperature": 0.3,
         "top_k": 2, "q_scaling": "none"}, NA)
    probs = policy.action_probs(Q, 0)
    assert probs.min() > 0.0
    np.testing.assert_allclose(probs[Q.argmin()], 0.1 / NA, atol=1e-12)


# ----------------------------------------------------------------- support

def test_top_k_support_sizes():
    for k in range(1, NA + 1):
        assert top_k_support(Q, k).sum() == k
    # Out-of-range k is clamped rather than raising: schedules produce
    # fractional values that round outside [1, |A|] at their endpoints.
    assert top_k_support(Q, 0).sum() == 1
    assert top_k_support(Q, 99).sum() == NA


def test_top_k_support_includes_boundary_ties():
    q = np.array([3.0, 2.0, 2.0, 1.0])
    # k=2 would cut through the tie at 2.0; both tied actions must be kept, or
    # the choice between them would be made by array order.
    assert top_k_support(q, 2).tolist() == [True, True, True, False]


def test_top_p_support_widens_when_q_is_flat():
    """Nucleus support must respond to how decided the Q-row is."""
    peaked = np.array([10.0, 0.0, 0.0, 0.0])
    flat = np.array([0.01, 0.0, 0.005, 0.002])
    assert top_p_support(peaked, 0.9, 1.0).sum() < top_p_support(flat, 0.9, 1.0).sum()


def test_anneal_k_walks_from_greedy_to_full_support():
    """The support path: k = 1 at the start, |A| at the end."""
    policy = build_policy(
        {"name": "anneal_k", "epsilon": 0.0, "temperature": 0.3,
         "top_k": {"type": "linear", "start": 1, "end": "num_actions",
                   "duration_frac": 1.0},
         "q_scaling": "none"}, NA, total_steps=1000)
    assert policy.knobs(0)[2] == 1
    assert policy.knobs(1000)[2] == NA
    # ... and at the start it is indistinguishable from epsilon-greedy.
    np.testing.assert_allclose(
        policy.action_probs(Q, 0), eps_greedy_reference(Q, 0.0), atol=1e-12)


def test_probabilities_always_normalised_across_the_whole_family():
    """Every registered variant, swept over its schedule, stays a distribution."""
    from e2b.policies import available_policies

    rng = np.random.default_rng(0)
    for name in available_policies():
        cfg: dict = {"name": name}
        if name == "topp_boltzmann":
            cfg["top_p"] = 0.9
        policy = build_policy(cfg, NA, total_steps=1000)
        for step in (0, 250, 500, 1000):
            q = rng.normal(size=NA) * rng.choice([1e-3, 1.0, 1e4])
            probs = policy.action_probs(q, step, uncertainty=0.5)
            assert np.all(np.isfinite(probs)), (name, step, probs)
            assert probs.min() >= -1e-15, (name, probs)
            assert abs(probs.sum() - 1.0) < 1e-9, (name, probs.sum())
