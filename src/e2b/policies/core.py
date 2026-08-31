"""The action-distribution algebra shared by every exploration variant.

Every strategy in this study is expressed as an explicit probability vector over
actions rather than as a procedure ("flip a coin, then argmax").  That costs a
few microseconds per step and buys three things:

* the limiting-case equivalences (top-k with k=1 is exactly epsilon-greedy,
  temperature -> 0 is exactly epsilon-greedy, ...) become *exact* array
  comparisons in the tests instead of flaky chi-square tests on samples;
* behaviour-policy entropy -- the key diagnostic for "has the handover from
  epsilon-greedy to Boltzmann actually happened?" -- is available for free;
* the mixture variant can interpolate two strategies exactly, rather than
  approximately via coin flips.

The single most important function here is :func:`masked_softmax`, whose
numerics are what make a temperature schedule that spans several orders of
magnitude actually usable.
"""

from __future__ import annotations

import numpy as np

# exp(-745) is the smallest double that does not underflow to exactly 0.
# Clipping the exponent here keeps `temperature -> 0` silent and exact instead
# of raising underflow warnings on every single step.
_MIN_EXPONENT = -700.0


def greedy_support(q: np.ndarray) -> np.ndarray:
    """Boolean mask of the argmax set (all tied maxima)."""
    return q >= q.max()


def top_k_support(q: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` highest-valued actions.

    Ties at the k-th boundary are *all* included.  This is what makes the
    limiting cases exact: with ``k = 1`` the support is the full argmax set, so
    the resulting distribution is uniform-over-argmax -- identical to what
    ``temperature -> 0`` over the full action set produces.  Had we broken ties
    by index instead, the two limits would disagree whenever Q has ties, which
    is exactly the situation at initialisation.
    """
    n = q.shape[-1]
    k = int(np.clip(k, 1, n))
    if k >= n:
        return np.ones_like(q, dtype=bool)
    # k-th largest value; partition is O(n) versus O(n log n) for a full sort.
    threshold = np.partition(q, n - k)[n - k]
    return q >= threshold


def top_p_support(q: np.ndarray, p: float, temperature: float) -> np.ndarray:
    """Nucleus support: the smallest high-Q action set with softmax mass >= ``p``.

    A state-adaptive alternative to a fixed ``k``: where Q is peaked the support
    collapses to one or two actions, and where Q is flat it widens.  That is the
    behaviour a fixed ``k`` cannot express, and it is the natural way to spend
    exploration only where the Q-function is genuinely undecided.
    """
    n = q.shape[-1]
    probs = masked_softmax(q, np.ones_like(q, dtype=bool), temperature)
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    # Number of actions needed to reach mass p (at least one).
    count = int(np.searchsorted(cumulative, min(p, 1.0), side="left")) + 1
    count = min(count, n)
    mask = np.zeros(n, dtype=bool)
    mask[order[:count]] = True
    return mask


def masked_softmax(
    q: np.ndarray, support: np.ndarray, temperature: float
) -> np.ndarray:
    """Boltzmann distribution over ``support``, zero elsewhere.

    Numerics: we subtract the in-support maximum *before* dividing by the
    temperature, so the exponent is always <= 0 and can only underflow, never
    overflow.  The naive ``exp(q / tau)`` overflows to inf for any large Q or
    small tau -- and both occur routinely, since Q grows during training while
    tau is being annealed down.

    In the ``temperature -> 0`` limit every non-maximal exponent underflows to
    0 and the maximal ones stay at ``exp(0) = 1``, so the result is exactly
    uniform over the in-support argmax set.  The limit is therefore reached
    exactly rather than approached, and no epsilon-thresholded special case is
    needed.
    """
    q = np.asarray(q, dtype=np.float64)
    masked = np.where(support, q, -np.inf)
    peak = masked.max()
    if not np.isfinite(peak):
        # Empty support should be impossible, but degrade to uniform rather
        # than emit NaNs into the action distribution.
        return np.ones_like(q) / q.shape[-1]

    temperature = max(float(temperature), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = (masked - peak) / temperature
    exponent = np.where(support, np.maximum(exponent, _MIN_EXPONENT), -np.inf)
    weights = np.exp(exponent)
    weights = np.where(support, weights, 0.0)
    total = weights.sum()
    if total <= 0.0:  # pragma: no cover - unreachable: the peak contributes 1.0
        return support.astype(np.float64) / support.sum()
    return weights / total


def compose_probs(
    q: np.ndarray,
    epsilon: float,
    temperature: float,
    support: np.ndarray,
) -> np.ndarray:
    """The one action distribution this whole project is about::

        pi(a) = eps * Uniform(A) + (1 - eps) * Softmax_{a in support}(q / tau)

    ``q`` is expected to be already scale-normalised (see
    :class:`e2b.policies.scaling.QScaler`); ``epsilon`` is a uniform floor over
    *all* actions, not just the support, so exploration can always reach an
    action that the current top-k has excluded.
    """
    n = q.shape[-1]
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    if epsilon >= 1.0:
        return uniform
    boltzmann = masked_softmax(q, support, temperature)
    probs = epsilon * uniform + (1.0 - epsilon) * boltzmann
    # Renormalise against accumulated float error so np.random.choice never
    # rejects the vector.
    return probs / probs.sum()


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats, with the usual 0 log 0 = 0 convention."""
    p = np.asarray(probs, dtype=np.float64)
    nonzero = p > 0.0
    return float(-(p[nonzero] * np.log(p[nonzero])).sum())


def non_greedy_mass(probs: np.ndarray, q: np.ndarray) -> float:
    """Probability the policy assigns to any non-argmax action.

    The headline "how much am I still exploring?" number, and the one that makes
    epsilon-greedy and Boltzmann directly comparable despite having completely
    different knobs.
    """
    return float(1.0 - probs[greedy_support(q)].sum())
