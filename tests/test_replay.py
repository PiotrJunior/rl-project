"""Replay: segment tree correctness, n-step returns, and PER sampling."""

import numpy as np
import pytest

from e2b.replay import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from e2b.replay.segment_tree import MinSegmentTree, SumSegmentTree


# ------------------------------------------------------------- segment tree

def test_sum_tree_prefix_search_matches_bruteforce():
    rng = np.random.default_rng(0)
    values = rng.random(16) * 10
    tree = SumSegmentTree(16)
    for i, v in enumerate(values):
        tree[i] = float(v)
    assert tree.sum() == pytest.approx(values.sum())

    cumulative = np.cumsum(values)
    queries = rng.uniform(0, values.sum(), size=200)
    got = tree.find_prefixsum_idx(queries)
    expected = np.searchsorted(cumulative, queries, side="right")
    np.testing.assert_array_equal(got, expected)


def test_sum_tree_sampling_frequency_tracks_priorities():
    """The property PER actually depends on."""
    tree = SumSegmentTree(4)
    priorities = [1.0, 3.0, 0.0, 4.0]
    for i, v in enumerate(priorities):
        tree[i] = v
    rng = np.random.default_rng(1)
    draws = tree.find_prefixsum_idx(rng.uniform(0, 8.0, size=100_000))
    freq = np.bincount(draws, minlength=4) / 100_000
    np.testing.assert_allclose(freq, np.array(priorities) / 8.0, atol=0.006)
    # A zero-priority transition must never be sampled.
    assert freq[2] == 0.0


def test_set_many_matches_individual_writes():
    a, b = SumSegmentTree(8), SumSegmentTree(8)
    values = np.arange(1, 9, dtype=np.float64)
    for i, v in enumerate(values):
        a[i], b[i] = float(v), float(v)
    idxs = np.array([0, 3, 3, 7])   # duplicates included on purpose
    new = np.array([9.0, 5.0, 6.0, 2.0])
    a.set_many(idxs, new)
    for i, v in zip(idxs, new):
        b[int(i)] = float(v)
    np.testing.assert_allclose(a.tree, b.tree)


def test_min_tree_tracks_minimum():
    tree = MinSegmentTree(8)
    for i, v in enumerate([5, 3, 9, 1, 7, 2, 8, 4]):
        tree[i] = float(v)
    assert tree.min() == 1.0
    tree[3] = 10.0
    assert tree.min() == 2.0


def test_capacity_must_be_power_of_two():
    with pytest.raises(ValueError):
        SumSegmentTree(10)


# ------------------------------------------------------------------- n-step

def test_n_step_return_matches_hand_computation():
    gamma = 0.9
    acc = NStepAccumulator(n_step=3, gamma=gamma)
    obs = [np.array([float(i)]) for i in range(5)]
    out = []
    for i in range(4):
        out.extend(acc.push(obs[i], i, float(i + 1), obs[i + 1], False, False))

    assert len(out) == 2
    first = out[0]
    # r0 + g*r1 + g^2*r2 = 1 + 0.9*2 + 0.81*3 = 5.23
    assert first.reward == pytest.approx(1 + gamma * 2 + gamma**2 * 3)
    assert first.discount == pytest.approx(gamma**3)
    assert first.obs[0] == 0.0 and first.next_obs[0] == 3.0
    assert not first.terminated


def test_n_step_truncates_at_terminal_and_does_not_cross_episodes():
    gamma = 0.9
    acc = NStepAccumulator(n_step=3, gamma=gamma)
    o = [np.array([float(i)]) for i in range(4)]
    out = []
    out.extend(acc.push(o[0], 0, 1.0, o[1], False, False))
    out.extend(acc.push(o[1], 1, 2.0, o[2], True, False))   # terminal here

    # Terminal drains the buffer: both partial transitions must be emitted.
    assert len(out) == 2
    assert out[0].reward == pytest.approx(1 + gamma * 2)
    assert out[0].terminated
    # The fold stops at the terminal, so the discount reflects k = 2, not n = 3.
    assert out[0].discount == pytest.approx(gamma**2)
    assert out[1].reward == pytest.approx(2.0)
    assert out[1].terminated


def test_n_step_flushes_on_truncation_without_marking_terminal():
    """A time-limit truncation must not be recorded as a true terminal.

    Treating it as terminal would tell the agent the value of the state is
    zero, which on CartPole (truncated at 500 steps) means punishing the agent
    for succeeding.
    """
    acc = NStepAccumulator(n_step=3, gamma=0.99)
    o = [np.array([float(i)]) for i in range(3)]
    out = list(acc.push(o[0], 0, 1.0, o[1], False, False))
    out += list(acc.push(o[1], 1, 1.0, o[2], False, True))
    assert out and all(not t.terminated for t in out)


def test_n_step_one_is_plain_one_step():
    acc = NStepAccumulator(n_step=1, gamma=0.99)
    out = list(acc.push(np.array([0.0]), 0, 2.0, np.array([1.0]), False, False))
    assert len(out) == 1
    assert out[0].reward == 2.0
    assert out[0].discount == pytest.approx(0.99)


# ------------------------------------------------------------------- buffers

def _fill(buf, n=50):
    rng = np.random.default_rng(0)
    for i in range(n):
        buf.add(rng.normal(size=3).astype(np.float32), i % 2, float(i),
                rng.normal(size=3).astype(np.float32), i % 7 == 0, 0.97)


def test_uniform_buffer_roundtrip_and_wraparound():
    buf = ReplayBuffer(capacity=16, obs_shape=(3,), rng=np.random.default_rng(0))
    _fill(buf, 40)
    assert len(buf) == 16          # capacity respected
    batch = buf.sample(8)
    assert len(batch) == 8
    assert batch.obs.shape == (8, 3)
    np.testing.assert_allclose(batch.weights, 1.0)


def test_prioritized_sampling_favours_high_priority_transitions():
    buf = PrioritizedReplayBuffer(
        capacity=8, obs_shape=(3,), alpha=1.0, rng=np.random.default_rng(0))
    _fill(buf, 8)
    # Make exactly one transition far more surprising than the rest.
    buf.update_priorities(np.arange(8), np.array([0.01] * 7 + [100.0]))
    counts = np.zeros(8)
    for _ in range(300):
        counts += np.bincount(buf.sample(8).indices, minlength=8)
    assert counts[7] > counts[:7].sum(), counts


def test_prioritized_importance_weights_are_bounded_and_inverse_to_priority():
    buf = PrioritizedReplayBuffer(
        capacity=8, obs_shape=(3,), alpha=1.0, rng=np.random.default_rng(0))
    _fill(buf, 8)
    buf.update_priorities(np.arange(8), np.linspace(0.1, 10.0, 8))
    batch = buf.sample(64, beta=1.0)
    assert batch.weights.max() <= 1.0 + 1e-6
    assert batch.weights.min() > 0.0
    # Higher-priority (over-sampled) transitions must get smaller weights.
    order = np.argsort(batch.indices)
    idx_sorted = batch.indices[order]
    w_sorted = batch.weights[order]
    if idx_sorted[0] != idx_sorted[-1]:
        assert w_sorted[0] >= w_sorted[-1]


def test_bootstrap_masks_are_binary_and_never_all_zero():
    """Every transition must be owned by at least one head, or it is dead weight."""
    buf = ReplayBuffer(capacity=64, obs_shape=(3,), num_heads=5,
                       bootstrap_prob=0.2, rng=np.random.default_rng(0))
    _fill(buf, 64)
    masks = buf.masks[: len(buf)]
    assert set(np.unique(masks)).issubset({0.0, 1.0})
    assert np.all(masks.sum(axis=1) >= 1)


def test_single_head_masks_are_all_ones():
    buf = ReplayBuffer(capacity=16, obs_shape=(3,), num_heads=1,
                       bootstrap_prob=0.5, rng=np.random.default_rng(0))
    _fill(buf, 16)
    np.testing.assert_allclose(buf.masks[: len(buf)], 1.0)


def test_sampling_empty_buffer_raises():
    buf = ReplayBuffer(capacity=8, obs_shape=(3,))
    with pytest.raises(ValueError):
        buf.sample(4)
