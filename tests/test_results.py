"""Loading run directories back into arrays."""

import json

import numpy as np

from e2b.results import ArmResults, group_by_env, load_run, load_sweep


def _write_run(run_dir, seed, returns, env="CartPole-v1", policy="eps_greedy"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(
        {"run_id": f"{env}/{policy}/seed{seed}", "env": env, "policy": policy,
         "seed": seed, "final_eval_return": returns[-1]}))
    lines = ["step,eval_return_mean,entropy"]
    for i, r in enumerate(returns, start=1):
        lines.append(f"{i * 100},{r},{0.5 / i}")
    (run_dir / "eval.csv").write_text("\n".join(lines) + "\n")
    (run_dir / "diagnostics.csv").write_text(
        "step,entropy,non_greedy\n" + "\n".join(
            f"{i * 50},{0.4 / i},{0.3 / i}" for i in range(1, 5)) + "\n")


def test_load_run_returns_none_for_an_incomplete_run(tmp_path):
    """A crashed or in-flight run must not silently contribute a partial curve."""
    (tmp_path / "eval.csv").write_text("step,eval_return_mean\n100,1.0\n")
    assert load_run(tmp_path) is None


def test_load_sweep_groups_seeds_into_arms(tmp_path):
    root = tmp_path / "sweep"
    for seed, returns in enumerate([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]):
        _write_run(root / "CartPole-v1" / "eps_greedy" / f"seed{seed}", seed, returns)
    arms = load_sweep(root)
    assert len(arms) == 1
    arm = arms[0]
    assert arm.label == "eps_greedy"
    assert arm.curves.shape == (2, 3)
    assert sorted(arm.seeds) == [0, 1]
    np.testing.assert_allclose(arm.steps, [100, 200, 300])


def test_load_sweep_truncates_to_the_shortest_seed(tmp_path):
    """Ragged curves must become a rectangular array, so every plotted step is
    backed by every seed rather than silently by a subset."""
    root = tmp_path / "sweep"
    _write_run(root / "CartPole-v1" / "eps_greedy" / "seed0", 0, [1.0, 2.0, 3.0, 4.0])
    _write_run(root / "CartPole-v1" / "eps_greedy" / "seed1", 1, [1.0, 2.0])
    arm = load_sweep(root)[0]
    assert arm.curves.shape == (2, 2)


def test_load_sweep_skips_incomplete_runs(tmp_path):
    root = tmp_path / "sweep"
    _write_run(root / "CartPole-v1" / "eps_greedy" / "seed0", 0, [1.0, 2.0])
    bad = root / "CartPole-v1" / "eps_greedy" / "seed1"
    bad.mkdir(parents=True)
    (bad / "eval.csv").write_text("step,eval_return_mean\n100,1.0\n")   # no result.json
    arm = load_sweep(root)[0]
    assert arm.curves.shape == (1, 2)


def test_load_sweep_of_missing_directory_is_empty(tmp_path):
    assert load_sweep(tmp_path / "nope") == []


def test_final_averages_a_tail_not_a_single_point():
    """One evaluation point is close to meaningless on a DQN curve."""
    arm = ArmResults(env="e", label="l", steps=np.arange(5),
                     curves=np.array([[0.0, 0.0, 3.0, 6.0, 9.0]]))
    assert arm.final[0] == 6.0          # mean of last three
    assert arm.auc[0] == 3.6            # mean of all five


def test_group_by_env_partitions_arms(tmp_path):
    root = tmp_path / "sweep"
    _write_run(root / "CartPole-v1" / "eps_greedy" / "seed0", 0, [1.0, 2.0])
    _write_run(root / "Acrobot-v1" / "eps_greedy" / "seed0", 0, [1.0, 2.0],
               env="Acrobot-v1")
    grouped = group_by_env(load_sweep(root))
    assert set(grouped) == {"CartPole-v1", "Acrobot-v1"}
