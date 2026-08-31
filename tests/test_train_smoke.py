"""End-to-end: every registered exploration variant must actually train.

Short runs (a few thousand steps) on CartPole. These will not learn anything
useful, and are not meant to -- they catch the class of bug that unit tests
cannot: a policy that works in isolation but crashes when wired to a real
environment, a config that does not compose, an arm that silently never
explores.
"""

from __future__ import annotations

import numpy as np
import pytest

from e2b.config import load_config
from e2b.train import train

POLICY_CONFIGS = [
    "policy/eps_greedy",
    "policy/boltzmann",
    "policy/eps_boltzmann",
    "policy/mixture_anneal",
    "policy/topk_boltzmann",
    "policy/topk3_boltzmann",
    "policy/anneal_k",
    "policy/topp_boltzmann",
    "policy/eps_boltzmann_none",
    "policy/eps_boltzmann_perstate",
    "policy/uncertainty_gated_td",
]

SHORT = [
    "train.total_steps=1200",
    "train.eval_interval=600",
    "train.eval_episodes=2",
    "train.diagnostics_interval=300",
    "agent.learning_starts=200",
    "agent.replay.capacity=2000",
    "agent.net.hidden_sizes=[32, 32]",
]


def _config(policy: str, tmp_path, extra=()):
    cfg = load_config(policy, [*SHORT, *extra])
    # `policy/*.yaml` files carry only a policy block, so compose them onto base.
    from e2b.config import config_from_dict, load_yaml_tree, _deep_merge

    data = _deep_merge(load_yaml_tree("base"), load_yaml_tree(policy))
    from e2b.config import apply_overrides

    data = apply_overrides(data, [*SHORT, *extra, f"run.out_dir={tmp_path}"])
    return config_from_dict(data)


@pytest.mark.parametrize("policy", POLICY_CONFIGS)
def test_every_variant_trains_end_to_end(policy, tmp_path):
    cfg = _config(policy, tmp_path)
    summary = train(cfg, tmp_path / "run")
    assert summary["total_steps"] == 1200
    assert np.isfinite(summary["final_eval_return"])
    assert summary["episodes"] > 0

    # The learning curve and the exploration diagnostics must both be populated;
    # an empty diagnostics file means the report cannot say anything about when
    # the epsilon-greedy -> Boltzmann handover happened.
    assert (tmp_path / "run" / "eval.csv").read_text().count("\n") >= 2
    assert (tmp_path / "run" / "diagnostics.csv").read_text().count("\n") >= 2


def test_ensemble_uncertainty_gated_trains_end_to_end(tmp_path):
    """The extension arm, which needs multi-head networks wired through."""
    cfg = _config("policy/uncertainty_gated", tmp_path,
                  extra=["agent.net.num_heads=3", "policy.warmup_steps=200"])
    summary = train(cfg, tmp_path / "run")
    assert np.isfinite(summary["final_eval_return"])

    import csv

    rows = list(csv.DictReader((tmp_path / "run" / "diagnostics.csv").open()))
    assert rows, "no diagnostics written"
    assert "confidence" in rows[0]
    # Confidence must be a real number in range, not a constant placeholder.
    values = [float(r["confidence"]) for r in rows if r["confidence"] != ""]
    assert values and all(0.0 <= v <= 1.0 for v in values)


def test_exploration_actually_differs_between_variants(tmp_path):
    """Guard against every arm collapsing to the same behaviour.

    If a wiring mistake made the policy ignore its config, every arm would still
    train fine and the whole study would silently compare identical agents. This
    asserts the behaviour policies genuinely differ, using entropy -- the one
    diagnostic that is comparable across strategies.
    """
    import csv

    def mean_entropy(policy: str) -> float:
        out = tmp_path / policy.replace("/", "_")
        cfg = _config(policy, out, extra=["run.seed=0"])
        train(cfg, out / "run")
        rows = list(csv.DictReader((out / "run" / "diagnostics.csv").open()))
        vals = [float(r["entropy"]) for r in rows if r.get("entropy")]
        return float(np.mean(vals))

    greedy_like = mean_entropy("policy/topk_boltzmann")   # eps decays fast, k=2
    boltzmann = mean_entropy("policy/boltzmann")          # no eps floor at all
    assert abs(greedy_like - boltzmann) > 1e-3


def test_ensemble_gating_with_one_head_is_rejected(tmp_path):
    """A single-head 'ensemble' has identically zero disagreement, so the gated
    policy would sit at confidence 0 and run as plain epsilon-greedy for the
    whole sweep -- a config that looks fine and quietly tests the wrong thing.
    """
    import pytest

    cfg = _config("policy/uncertainty_gated", tmp_path,
                  extra=["agent.net.num_heads=1"])
    with pytest.raises(ValueError, match="num_heads >= 2"):
        train(cfg, tmp_path / "run")
