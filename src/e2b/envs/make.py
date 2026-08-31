"""Environment construction and the observation contract the agent relies on."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from e2b.config import EnvConfig


@dataclass
class EnvSpec:
    """What the agent needs to know about an environment to build itself."""

    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype
    num_actions: int
    net_kind: str  # "mlp" for state vectors, "cnn" for stacked frames


def make_env(cfg: EnvConfig, seed: int | None = None, render_mode: str | None = None):
    """Build a single environment from config."""
    if cfg.kind == "atari":
        from e2b.envs.atari_wrappers import make_atari_env

        return make_atari_env(
            cfg.id,
            seed=seed,
            frame_skip=cfg.frame_skip,
            frame_stack=cfg.frame_stack,
            screen_size=cfg.screen_size,
            grayscale=cfg.grayscale,
            terminal_on_life_loss=cfg.terminal_on_life_loss,
            repeat_action_probability=cfg.repeat_action_probability,
            clip_rewards=cfg.clip_rewards,
            noop_max=cfg.noop_max,
            render_mode=render_mode,
        )

    kwargs: dict[str, Any] = {}
    if cfg.max_episode_steps is not None:
        kwargs["max_episode_steps"] = cfg.max_episode_steps
    env = gym.make(cfg.id, render_mode=render_mode, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def describe_env(env) -> EnvSpec:
    """Derive the agent-facing spec from a constructed environment."""
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError(
            f"this project studies discrete-action exploration; got {env.action_space}"
        )
    obs_space = env.observation_space
    shape = tuple(obs_space.shape)
    dtype = np.dtype(obs_space.dtype)

    if len(shape) == 1:
        return EnvSpec(shape, np.dtype(np.float32), int(env.action_space.n), "mlp")
    if len(shape) == 3:
        # Frame-stacked Atari arrives as (stack, H, W) from FrameStackObservation
        # over grayscale frames -- already channel-first, so no transpose here.
        return EnvSpec(shape, dtype, int(env.action_space.n), "cnn")
    raise ValueError(f"unsupported observation shape {shape}")
