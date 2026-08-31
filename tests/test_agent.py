"""Agent-level correctness: Double DQN targets, n-step bootstrapping, ensembles."""

import numpy as np
import pytest
import torch

from e2b.agent import DQNAgent
from e2b.config import AgentConfig, NetConfig, ReplayConfig
from e2b.envs.make import EnvSpec
from e2b.nets import build_qnetwork


def make_agent(**net_kwargs):
    # Seed torch too, not just the agent's numpy stream: network initialisation
    # comes from torch's global RNG, so without this the agent under test is a
    # different network on every run. That made
    # test_learn_runs_and_reduces_loss_on_a_fixed_batch flaky (~1 run in 5).
    torch.manual_seed(0)
    cfg = AgentConfig(
        batch_size=8, learning_starts=8, target_update_interval=10**9,
        net=NetConfig(hidden_sizes=(16,), **net_kwargs),
        replay=ReplayConfig(capacity=64, prioritized=True, n_step=1),
    )
    spec = EnvSpec((4,), np.dtype(np.float32), 3, "mlp")
    return DQNAgent(cfg, spec, torch.device("cpu"), np.random.default_rng(0),
                    total_steps=1000)


def test_network_output_shape_carries_head_axis():
    net = build_qnetwork((4,), 3, hidden_sizes=(16,), num_heads=4)
    out = net(torch.zeros(5, 4))
    assert out.shape == (5, 4, 3)
    assert net.q_values(torch.zeros(5, 4)).shape == (5, 3)


def test_single_head_still_has_head_axis():
    """Keeps the agent and the uncertainty estimators branch-free."""
    net = build_qnetwork((4,), 3, hidden_sizes=(16,), num_heads=1)
    assert net(torch.zeros(2, 4)).shape == (2, 1, 3)


def test_dueling_head_is_mean_centred():
    net = build_qnetwork((4,), 3, hidden_sizes=(16,), dueling=True)
    head = net.heads[0]
    features = torch.randn(7, net.torso.out_dim)
    q = head(features)
    advantage = head.advantage(features)
    value = head.value(features)
    expected = value + advantage - advantage.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(q, expected)


def test_double_dqn_target_differs_from_vanilla_when_argmaxes_disagree():
    """Constructed case: online and target nets prefer different actions.

    If this ever collapses to equality, the Double-DQN wiring has silently
    reverted to a plain max over the target network.
    """
    agent = make_agent()
    with torch.no_grad():
        # Force online to prefer action 0 and target to prefer action 2, with
        # very different values, so the two targets cannot coincide.
        for net, bias in ((agent.online, [5.0, 0.0, 0.0]), (agent.target, [0.0, 0.0, 9.0])):
            head = net.heads[0]
            head.advantage.weight.zero_()
            head.advantage.bias.copy_(torch.tensor(bias))
            head.value.weight.zero_()
            head.value.bias.zero_()

    t = {
        "next_obs": torch.zeros(1, 4),
        "rewards": torch.zeros(1),
        "terminated": torch.zeros(1),
        "discounts": torch.full((1,), 0.99),
    }
    agent.cfg.double_dqn = True
    double = agent._compute_targets(t).item()
    agent.cfg.double_dqn = False
    vanilla = agent._compute_targets(t).item()

    # The dueling head mean-centres the advantage, so with advantage biases
    # [0, 0, 9] the target net's Q is [-3, -3, 6], and with [5, 0, 0] the online
    # net's Q is [10/3, -5/3, -5/3] (argmax = action 0).
    # Vanilla takes the target net's own max  -> 0.99 * 6  =  5.94
    # Double evaluates the online net's pick  -> 0.99 * -3 = -2.97
    assert vanilla > double
    assert double == pytest.approx(0.99 * -3.0, abs=1e-4)
    assert vanilla == pytest.approx(0.99 * 6.0, abs=1e-4)


def test_terminal_transitions_do_not_bootstrap():
    agent = make_agent()
    t = {
        "next_obs": torch.zeros(2, 4),
        "rewards": torch.tensor([1.0, 1.0]),
        "terminated": torch.tensor([0.0, 1.0]),
        "discounts": torch.tensor([0.99, 0.99]),
    }
    targets = agent._compute_targets(t)
    assert targets[1].item() == pytest.approx(1.0)   # reward only
    assert targets[0].item() != pytest.approx(1.0)   # bootstrapped


def test_target_uses_per_transition_discount_not_fixed_gamma_n():
    """n-step transitions flushed at an episode boundary have k < n.

    Using a fixed gamma**n for them would over-discount the bootstrap term.
    """
    agent = make_agent()
    base = {
        "next_obs": torch.zeros(2, 4),
        "rewards": torch.zeros(2),
        "terminated": torch.zeros(2),
    }
    a = agent._compute_targets({**base, "discounts": torch.tensor([0.9, 0.9])})
    b = agent._compute_targets({**base, "discounts": torch.tensor([0.9, 0.5])})
    assert a[0].item() == pytest.approx(b[0].item())
    assert a[1].item() != pytest.approx(b[1].item())


def test_learn_returns_none_until_learning_starts():
    agent = make_agent()
    assert agent.learn(0) is None


def test_learn_runs_and_reduces_loss_on_a_fixed_batch():
    """Sanity check that gradients actually flow through the whole stack."""
    agent = make_agent()
    rng = np.random.default_rng(0)
    for i in range(64):
        agent.observe(rng.normal(size=4).astype(np.float32), i % 3, 1.0,
                      rng.normal(size=4).astype(np.float32), False, False)
    first = agent.learn(0)
    assert first is not None
    losses = [first["loss"]]
    for _ in range(200):
        losses.append(agent.learn(0)["loss"])
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_ensemble_heads_disagree_at_initialisation():
    """The uncertainty signal depends on heads being independently initialised."""
    agent = make_agent(num_heads=5)
    q = agent.q_heads(np.zeros(4, dtype=np.float32))
    assert q.shape == (5, 3)
    assert q.std(axis=0).mean() > 0.0


def test_q_values_is_head_mean():
    agent = make_agent(num_heads=4)
    obs = np.zeros(4, dtype=np.float32)
    np.testing.assert_allclose(
        agent.q_values(obs), agent.q_heads(obs).mean(axis=0), rtol=1e-6)


def test_bootstrap_masks_restrict_which_heads_learn():
    """A head that owns no samples must receive no gradient.

    This is what makes the heads' disagreement meaningful rather than a
    difference in initialisation that decays to nothing.
    """
    agent = make_agent(num_heads=3)
    rng = np.random.default_rng(0)
    for i in range(64):
        agent.observe(rng.normal(size=4).astype(np.float32), i % 3, 1.0,
                      rng.normal(size=4).astype(np.float32), False, False)
    # Mask every stored transition to head 0 only.
    agent.replay.masks[:] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    before = [h.advantage.weight.detach().clone() for h in agent.online.heads]
    agent.learn(0)
    after = [h.advantage.weight.detach() for h in agent.online.heads]
    assert not torch.allclose(before[0], after[0])      # head 0 learned
    torch.testing.assert_close(before[1], after[1])     # heads 1, 2 untouched
    torch.testing.assert_close(before[2], after[2])


def test_priorities_are_updated_after_learning():
    agent = make_agent()
    rng = np.random.default_rng(0)
    for i in range(64):
        agent.observe(rng.normal(size=4).astype(np.float32), i % 3, float(i),
                      rng.normal(size=4).astype(np.float32), False, False)
    tree = agent.replay._sum_tree
    before = tree.sum()
    agent.learn(0)
    assert tree.sum() != before


def test_beta_anneals_from_start_to_end():
    agent = make_agent()
    assert agent.beta(0) == pytest.approx(agent.cfg.replay.beta_start)
    assert agent.beta(1000) == pytest.approx(agent.cfg.replay.beta_end)
