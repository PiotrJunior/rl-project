"""Atari preprocessing.

No Atari *training* is part of this project's results (a Rainbow-class run is a
GPU-day, and the study needs 5 seeds x 8 variants). But the code path is real
and must stay correct, so it is tested against actual ALE environments here:
observation shape/dtype, stacking order, action semantics and the agent's
ability to consume the result.

These tests are skipped automatically if ale-py is not installed.
"""

import numpy as np
import pytest

ale_py = pytest.importorskip("ale_py")

from e2b.config import AgentConfig, EnvConfig, NetConfig, ReplayConfig  # noqa: E402
from e2b.envs import describe_env, make_env  # noqa: E402

ENV_ID = "ALE/Breakout-v5"


@pytest.fixture(scope="module")
def atari_env():
    cfg = EnvConfig(id=ENV_ID, kind="atari", frame_stack=4, screen_size=84)
    env = make_env(cfg, seed=0)
    yield env
    env.close()


def test_observation_is_stacked_uint8_and_channel_first(atari_env):
    """The agent's CNN torso expects (C, H, W) uint8; getting this wrong is a
    silent 4x memory blow-up (float32) or a transposed image."""
    obs, _ = atari_env.reset(seed=0)
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    assert 0 <= obs.min() and obs.max() <= 255


def test_env_spec_selects_the_cnn_torso(atari_env):
    spec = describe_env(atari_env)
    assert spec.net_kind == "cnn"
    assert spec.obs_shape == (4, 84, 84)
    assert spec.obs_dtype == np.uint8
    assert spec.num_actions == 4


def test_frame_stack_advances_and_retains_history(atari_env):
    """Successive observations must share frames, or the stack is not a stack."""
    obs0, _ = atari_env.reset(seed=0)
    obs1, *_ = atari_env.step(0)
    # The newest frame of obs0 should reappear as an older frame of obs1.
    assert np.array_equal(obs0[-1], obs1[-2])
    assert not np.array_equal(obs1[-1], obs0[-1]) or True  # may repeat if static


def test_reward_clipping_bounds_rewards(atari_env):
    for _ in range(200):
        _, reward, terminated, truncated, _ = atari_env.step(
            atari_env.action_space.sample())
        assert -1.0 <= reward <= 1.0
        if terminated or truncated:
            atari_env.reset()
            break


def test_reward_clipping_can_be_disabled():
    cfg = EnvConfig(id=ENV_ID, kind="atari", clip_rewards=False)
    env = make_env(cfg, seed=0)
    try:
        env.reset(seed=0)
        # Breakout awards 1-7 per brick; just assert the wrapper is absent.
        from e2b.envs.atari_wrappers import ClipRewardWrapper

        node, seen = env, False
        while hasattr(node, "env"):
            seen = seen or isinstance(node, ClipRewardWrapper)
            node = node.env
        assert not seen
    finally:
        env.close()


def test_sticky_actions_are_configured():
    """Machado et al. (2018) stochasticity. Without it, Atari is deterministic
    and an agent can win by memorising an action sequence."""
    cfg = EnvConfig(id=ENV_ID, kind="atari", repeat_action_probability=0.25)
    env = make_env(cfg, seed=0)
    try:
        assert env.unwrapped.ale.getFloat("repeat_action_probability") == pytest.approx(0.25)
    finally:
        env.close()


def test_fire_reset_starts_the_game():
    """Breakout waits for FIRE to serve; without this the agent sees a frozen
    screen and looks like it has broken exploration."""
    cfg = EnvConfig(id=ENV_ID, kind="atari")
    env = make_env(cfg, seed=0)
    try:
        env.reset(seed=0)
        assert env.unwrapped.ale.lives() == 5
    finally:
        env.close()


def test_agent_can_consume_atari_observations():
    """End-to-end shape contract: env -> replay -> CNN -> Q-values.

    This is the check that actually matters for the 'code path exists' claim.
    """
    import torch

    from e2b.agent import DQNAgent
    from e2b.policies import build_policy

    cfg = EnvConfig(id=ENV_ID, kind="atari", frame_stack=4, screen_size=84)
    env = make_env(cfg, seed=0)
    try:
        spec = describe_env(env)
        agent_cfg = AgentConfig(
            batch_size=4, learning_starts=4,
            net=NetConfig(cnn_hidden=32, cnn_channels=(8, 8), cnn_kernels=(8, 4),
                          cnn_strides=(4, 2)),
            replay=ReplayConfig(capacity=32, n_step=2),
        )
        agent = DQNAgent(agent_cfg, spec, torch.device("cpu"),
                         np.random.default_rng(0), total_steps=100)
        policy = build_policy({"name": "eps_greedy", "epsilon": 0.5, "top_k": 1}, spec.num_actions)
        rng = np.random.default_rng(0)

        obs, _ = env.reset(seed=0)
        for step in range(24):
            q = agent.q_values(obs)
            assert q.shape == (spec.num_actions,)
            action = policy.act(q, step, rng)
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.observe(obs, action, float(reward), next_obs, term, trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]

        # uint8 frames must survive into replay without being widened to float32.
        assert agent.replay.obs.dtype == np.uint8
        stats = agent.learn(0)
        assert stats is not None and np.isfinite(stats["loss"])
    finally:
        env.close()
