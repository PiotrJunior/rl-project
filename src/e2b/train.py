"""Single training run: one config + one seed.

The loop is deliberately plain. Everything that varies between experimental
arms lives in the exploration policy; the agent, the optimiser, the replay and
the schedule of gradient steps are identical across arms by construction.

Diagnostics are the reason this file is longer than a minimal DQN loop. To say
anything about *when* the epsilon-greedy -> Boltzmann handover happens, and
whether it is the handover rather than something else that moved performance,
the run has to record the exploration policy's internal state alongside the
learning curve -- see :func:`_diagnostic_row`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e2b.agent import DQNAgent
from e2b.config import Config, config_to_dict, dump_config, load_config
from e2b.envs import describe_env, make_env
from e2b.evaluate import evaluate
from e2b.policies import build_policy, requires_uncertainty
from e2b.uncertainty.estimators import TdErrorUncertainty
from e2b.utils.logging import RunLogger
from e2b.utils.seeding import make_rng, resolve_device, seed_everything


class DiagnosticAccumulator:
    """Averages the exploration policy's per-step diagnostics over an interval.

    Logging the policy's state at a single sampled step produces a signal
    dominated by which state happened to be visited at that instant -- entropy
    at one state says nothing about the behaviour policy overall. Since the
    point of these columns is to show *when the epsilon-greedy -> Boltzmann
    handover actually happened*, they have to be averages over the interval,
    not snapshots taken at its edge.

    Scheduled knobs (epsilon, temperature) are deterministic functions of the
    step, so averaging them is harmless; state-dependent quantities (entropy,
    non-greedy mass, support size, confidence) are the ones that need it.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not np.isfinite(value):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._counts[key] = self._counts.get(key, 0) + 1

    def mean(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k]}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def _diagnostic_row(step: int, accumulator, agent, train_stats, recent_returns) -> dict:
    """Interval-averaged exploration internals + learning state.

    ``entropy`` and ``non_greedy`` are the load-bearing columns: they are
    comparable across every variant, whereas epsilon and temperature are not
    (an epsilon of 0.05 and a temperature of 0.3 are not "the same amount" of
    exploration, and how much exploration a given temperature buys drifts as
    the Q-scale grows).
    """
    row: dict[str, Any] = {"step": step}
    row.update(accumulator.mean())
    if train_stats:
        row.update(train_stats)
    row["replay_size"] = len(agent.replay)
    if recent_returns:
        row["train_return_mean"] = float(np.mean(recent_returns))
    return row


def train(cfg: Config, run_dir: Path | None = None) -> dict[str, Any]:
    """Execute one run and return its summary."""
    torch.set_num_threads(max(1, cfg.run.torch_threads))

    seed = cfg.run.seed
    seed_everything(seed)
    device = resolve_device(cfg.run.device)

    # Separate RNG streams: exploration must not shift when the number of
    # network updates changes, and evaluation must not perturb either.
    action_rng = make_rng(seed * 7919 + 1)
    replay_rng = make_rng(seed * 7919 + 2)

    env = make_env(cfg.env, seed=seed)
    eval_env = make_env(cfg.env, seed=seed + 10_000)
    spec = describe_env(env)

    agent = DQNAgent(
        cfg.agent, spec, device, replay_rng, total_steps=cfg.train.total_steps
    )
    policy = build_policy(cfg.policy, spec.num_actions, cfg.train.total_steps)
    needs_uncertainty = requires_uncertainty(cfg.policy)

    # The td_error signal is fed from the training loop rather than the acting
    # path, so it needs a handle here.
    td_signal = (
        policy.estimator
        if needs_uncertainty and isinstance(getattr(policy, "estimator", None), TdErrorUncertainty)
        else None
    )

    run_dir = Path(run_dir or Path(cfg.run.out_dir) / cfg.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    logger = RunLogger(run_dir)

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    recent_returns: deque[float] = deque(maxlen=20)
    train_stats: dict[str, float] | None = None
    curve: list[dict[str, Any]] = []
    diagnostics = DiagnosticAccumulator()
    started = time.time()

    progress = None
    if cfg.train.progress:
        from tqdm import tqdm

        progress = tqdm(total=cfg.train.total_steps, desc=cfg.run_id, unit="step")

    for step in range(cfg.train.total_steps):
        q_heads = agent.q_heads(obs)
        q = q_heads.mean(axis=0)

        confidence = None
        if needs_uncertainty:
            if td_signal is not None:
                confidence = td_signal.confidence()
            else:
                confidence = policy.estimator.confidence(q_heads=q_heads)
                # The reference quantile for the ensemble signal is updated from
                # the acting path, since that is where it is measured.
                policy.estimator.update_reference()

        action = policy.act(q, step, action_rng, uncertainty=confidence)
        diagnostics.add(policy.diagnostics())
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.observe(obs, action, float(reward), next_obs, terminated, truncated)
        episode_return += float(reward)
        episode_length += 1
        obs = next_obs

        if terminated or truncated:
            recent_returns.append(episode_return)
            logger.episodes.write(
                {
                    "step": step + 1,
                    "episode": episode_index,
                    "return": episode_return,
                    "length": episode_length,
                }
            )
            episode_index += 1
            episode_return = 0.0
            episode_length = 0
            obs, _ = env.reset()

        if step % max(1, cfg.agent.train_frequency) == 0:
            stats = agent.learn(step)
            if stats is not None:
                train_stats = stats
                if td_signal is not None:
                    td_signal.observe_td_errors(np.array([stats["td_error"]]))

        if (step + 1) % max(1, cfg.train.diagnostics_interval) == 0:
            logger.diagnostics.write(
                _diagnostic_row(step + 1, diagnostics, agent, train_stats, recent_returns)
            )
            diagnostics.reset()

        if (step + 1) % max(1, cfg.train.eval_interval) == 0 or step + 1 == cfg.train.total_steps:
            metrics = evaluate(
                agent, eval_env, cfg.train.eval_episodes, seed=seed + 50_000
            )
            row = {"step": step + 1, **metrics}
            row["train_return_mean"] = (
                float(np.mean(recent_returns)) if recent_returns else float("nan")
            )
            row.update(
                {
                    k: v
                    for k, v in policy.diagnostics().items()
                    if k in ("eps", "temperature", "top_k", "entropy", "non_greedy",
                             "confidence", "q_scale")
                }
            )
            logger.eval.write(row)
            curve.append(row)

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    elapsed = time.time() - started
    returns = [row["eval_return_mean"] for row in curve]
    # "Final" performance averages the last few evaluation points rather than
    # taking the single last one: DQN learning curves are noisy enough that one
    # evaluation point is close to meaningless as a summary statistic.
    tail = returns[-3:] if len(returns) >= 3 else returns
    summary = {
        "run_id": cfg.run_id,
        "env": cfg.env.id,
        "policy": cfg.policy.get("name"),
        "seed": seed,
        "total_steps": cfg.train.total_steps,
        "final_eval_return": float(np.mean(tail)) if tail else float("nan"),
        "best_eval_return": float(np.max(returns)) if returns else float("nan"),
        "auc_eval_return": float(np.mean(returns)) if returns else float("nan"),
        "episodes": episode_index,
        "wall_time_s": elapsed,
        "steps_per_second": cfg.train.total_steps / max(elapsed, 1e-9),
    }
    logger.write_result(summary)
    logger.close()
    env.close()
    eval_env.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one DQN run.")
    parser.add_argument("--config", required=True, help="config name, e.g. experiments/smoke")
    parser.add_argument("--set", nargs="*", default=[], help="overrides key.path=value")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    summary = train(cfg, Path(args.run_dir) if args.run_dir else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
