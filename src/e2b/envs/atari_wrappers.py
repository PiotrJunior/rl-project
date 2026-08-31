"""Standard Atari preprocessing.

This is the **Atari code path**: it is implemented and unit-tested, but no Atari
training runs are part of this project's results.  A single Rainbow-class Atari
run is ~10M frames, which is a GPU-day; the study here is on Gym control tasks.
The path exists so the same agent and the same exploration policies can be
pointed at ALE without modification -- see ``configs/env/atari.yaml`` and the
"Atari" section of the README.

The preprocessing follows Machado et al. (2018) "Revisiting the ALE": sticky
actions rather than random no-ops as the source of stochasticity, no
loss-of-life episode termination by default, and full action set left to the
config.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np

try:  # pragma: no cover - exercised only when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class FireResetWrapper(gym.Wrapper if gym else object):  # type: ignore[misc]
    """Press FIRE after reset in games that need it to start.

    Without this, agents on Breakout-likes spend the early training doing
    nothing while the environment waits for a serve, which reads as a broken
    exploration strategy when it is really a broken reset.
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self._fire_index = meanings.index("FIRE") if "FIRE" in meanings else None

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        if self._fire_index is None:
            return obs, info
        obs, _, terminated, truncated, info = self.env.step(self._fire_index)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeWrapper(gym.Wrapper if gym else object):  # type: ignore[misc]
    """Treat loss of life as episode end for the *agent*, not for the emulator.

    Helps value propagation, but changes the semantics of "episode", so the
    training-return logging must use the underlying game score. Off by default
    (Machado et al. recommend against it for benchmark comparability).
    """

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._lives = 0
        self._real_done = True

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self._lives:
            terminated = True
        self._lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any):
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Continue from the current emulator state: only the agent-visible
            # episode ended, not the game.
            obs, _, _, _, info = self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper if gym else object):  # type: ignore[misc]
    """Sign-clip rewards, as in the original DQN.

    Note for this project: reward clipping compresses the Q-value scale, which
    interacts directly with Boltzmann temperature selection. The ``running``
    Q-scaler makes the temperature invariant to it, which is one reason that
    scaler is the default -- see ``e2b.policies.scaling``.
    """

    def reward(self, reward: SupportsFloat) -> float:
        return float(np.sign(float(reward)))


class TransposeImageWrapper(gym.ObservationWrapper if gym else object):  # type: ignore[misc]
    """(H, W, C) -> (C, H, W) for torch conv layers."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, (2, 0, 1))


def make_atari_env(
    env_id: str,
    seed: int | None = None,
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_size: int = 84,
    grayscale: bool = True,
    terminal_on_life_loss: bool = False,
    repeat_action_probability: float = 0.25,
    clip_rewards: bool = True,
    noop_max: int = 30,
    render_mode: str | None = None,
) -> Any:
    """Build a fully preprocessed ALE environment.

    Frame skipping is delegated to the ALE itself (``frameskip=1`` on the base
    env plus ``AtariPreprocessing``), which also performs the max-pooling over
    the last two frames needed to handle the console's alternating-sprite
    flicker.
    """
    import ale_py  # noqa: F401  (registers the ALE/* environment ids)
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=repeat_action_probability,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,  # applied below so ordering is explicit
        grayscale_obs=grayscale,
        scale_obs=False,  # keep uint8: 4x less replay memory than float32
    )
    if terminal_on_life_loss:
        env = EpisodicLifeWrapper(env)
    env = FireResetWrapper(env)
    if clip_rewards:
        env = ClipRewardWrapper(env)
    env = FrameStackObservation(env, stack_size=frame_stack)
    if not grayscale:
        env = TransposeImageWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
