"""Greedy evaluation, used for every learning curve in the study.

Evaluation is **always greedy**, for every exploration variant, in a separate
environment with its own seed stream. This is the only way the comparison is
meaningful: a Boltzmann behaviour policy scores differently from an
epsilon-greedy one *for reasons that have nothing to do with how well it
learned*, so measuring the behaviour policy's own return would conflate
exploration cost with learning quality. What we want to know is how good the
greedy policy extracted from Q is, as a function of the exploration that
produced it.
"""

from __future__ import annotations

import numpy as np


def evaluate(agent, env, episodes: int = 10, seed: int | None = None) -> dict[str, float]:
    """Run ``episodes`` greedy episodes and summarise the returns.

    Ties in the argmax are broken uniformly at random rather than by index:
    index tie-breaking silently biases evaluation towards low-numbered actions,
    which for an under-trained network (where Q rows are near-constant) can look
    like a real policy difference between arms.
    """
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []

    for episode in range(episodes):
        reset_seed = None if seed is None else int(seed + episode)
        obs, _ = env.reset(seed=reset_seed)
        done = False
        total = 0.0
        steps = 0
        while not done:
            q = agent.q_values(obs)
            best = np.flatnonzero(q >= q.max())
            action = int(rng.choice(best))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "eval_return_mean": float(arr.mean()),
        "eval_return_std": float(arr.std()),
        "eval_return_min": float(arr.min()),
        "eval_return_max": float(arr.max()),
        "eval_length_mean": float(np.mean(lengths)),
    }
