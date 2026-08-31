"""Every shipped sweep config must expand into valid, distinct runs.

A broken sweep config otherwise only surfaces partway through a multi-hour run,
which on this project's compute budget is expensive to discover late.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sweep import build_cells, cell_run_dir  # noqa: E402

from e2b.config import config_from_dict  # noqa: E402
from e2b.policies import build_policy  # noqa: E402

SWEEP_ROOT = ROOT / "configs" / "sweeps"
SWEEPS = sorted(p.stem for p in SWEEP_ROOT.glob("*.yaml"))


def _cells(name: str):
    spec = yaml.safe_load((SWEEP_ROOT / f"{name}.yaml").read_text())
    return spec, build_cells(spec)


def test_there_are_sweeps_to_check():
    assert SWEEPS, "no sweep configs found"


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_expands_into_valid_configs(name):
    spec, cells = _cells(name)
    assert cells, f"sweep {name} expanded to zero runs"
    for data in cells:
        cfg = config_from_dict({k: v for k, v in data.items() if k != "_label"})
        assert cfg.train.total_steps > 0
        assert cfg.agent.learning_starts < cfg.train.total_steps


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_policies_are_constructible(name):
    """Catches an unknown policy name or a bad schedule spec before launch."""
    _, cells = _cells(name)
    for data in cells:
        cfg = config_from_dict({k: v for k, v in data.items() if k != "_label"})
        policy = build_policy(cfg.policy, num_actions=4,
                              total_steps=cfg.train.total_steps)
        assert policy.num_actions == 4


@pytest.mark.parametrize("name", SWEEPS)
def test_sweep_run_directories_are_unique(name):
    """Two arms colliding on one directory would silently overwrite each other.

    This is a live risk: several policy config FILES share a policy `name`
    (topk_boltzmann.yaml and topk3_boltzmann.yaml are both the `topk_boltzmann`
    policy at different k), so labels are derived from the file stem.
    """
    _, cells = _cells(name)
    dirs = []
    for data in cells:
        label = data["_label"]
        cfg = config_from_dict({k: v for k, v in data.items() if k != "_label"})
        dirs.append(cell_run_dir(cfg, label, Path("results")))
    assert len(set(dirs)) == len(dirs), f"duplicate run dirs in sweep {name}"


@pytest.mark.parametrize("name", SWEEPS)
def test_all_arms_share_identical_agent_settings_except_declared_overrides(name):
    """The study's core assumption: only the exploration policy varies.

    Any agent-level difference between arms must come from an explicit
    per-arm `overrides` block in the sweep (as the uncertainty study does for
    ensemble head counts), never by accident from a policy config.
    """
    spec, cells = _cells(name)
    declared = {
        key
        for arm in (spec.get("arms") or [])
        for key in (arm.get("overrides", {}) or {}).get("agent", {})
    }
    baseline = None
    for data in cells:
        cfg = config_from_dict({k: v for k, v in data.items() if k != "_label"})
        agent = {
            "gamma": cfg.agent.gamma, "lr": cfg.agent.lr,
            "batch_size": cfg.agent.batch_size,
            "train_frequency": cfg.agent.train_frequency,
            "target_update_interval": cfg.agent.target_update_interval,
            "n_step": cfg.agent.replay.n_step,
            "prioritized": cfg.agent.replay.prioritized,
        }
        if "net" not in declared:
            agent["hidden_sizes"] = cfg.agent.net.hidden_sizes
            agent["num_heads"] = cfg.agent.net.num_heads
            agent["dueling"] = cfg.agent.net.dueling
        if baseline is None:
            baseline = agent
        assert agent == baseline, (
            f"sweep {name}: arm {data['_label']} differs from the shared agent "
            f"config in a way the sweep did not declare"
        )


def test_uncertainty_sweep_has_a_matched_architecture_control():
    """Without it, the extension's result is confounded with the ensemble
    architecture rather than attributable to the gating."""
    spec, _ = _cells("uncertainty")
    labels = {arm["label"] for arm in spec["arms"]}
    assert "gated_ensemble" in labels
    assert "eps_greedy_ensemble" in labels
    heads = {
        arm["label"]: arm.get("overrides", {}).get("agent", {}).get("net", {}).get("num_heads")
        for arm in spec["arms"]
    }
    assert heads["gated_ensemble"] == heads["eps_greedy_ensemble"] == 5
