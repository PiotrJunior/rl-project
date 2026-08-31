"""Config composition, override precedence, and typo rejection."""

import pytest

from e2b.config import apply_overrides, config_from_dict, load_config, load_yaml_tree


def test_defaults_compose_with_file_keys_winning():
    data = load_yaml_tree("experiments/smoke")
    # from base.yaml
    assert data["agent"]["gamma"] == 0.99
    # from env/cartpole.yaml
    assert data["env"]["id"] == "CartPole-v1"
    # the experiment file's own key overrides the env default
    assert data["train"]["total_steps"] == 3000


def test_policy_block_is_replaced_not_merged_when_name_changes():
    """Two variants have disjoint knobs, so inheriting leftovers would
    mis-configure the new policy (e.g. a stale `top_k: 1` would turn a
    Boltzmann arm back into epsilon-greedy without any error)."""
    base = {"policy": {"name": "eps_greedy", "top_k": 1, "epsilon": 0.1}}
    from e2b.config import _deep_merge

    merged = _deep_merge(base, {"policy": {"name": "boltzmann", "temperature": 0.5}})
    assert merged["policy"] == {"name": "boltzmann", "temperature": 0.5}
    assert "top_k" not in merged["policy"]


def test_same_policy_name_still_merges_keywise():
    from e2b.config import _deep_merge

    base = {"policy": {"name": "eps_greedy", "top_k": 1, "epsilon": 0.1}}
    merged = _deep_merge(base, {"policy": {"name": "eps_greedy", "epsilon": 0.5}})
    assert merged["policy"] == {"name": "eps_greedy", "top_k": 1, "epsilon": 0.5}


def test_cli_overrides_parse_yaml_values():
    data = apply_overrides({}, ["train.total_steps=1000", "run.device=cpu",
                                "agent.net.hidden_sizes=[64, 64]"])
    assert data["train"]["total_steps"] == 1000
    assert data["run"]["device"] == "cpu"
    assert data["agent"]["net"]["hidden_sizes"] == [64, 64]


def test_unknown_key_is_rejected():
    """A typo must fail loudly rather than produce a run that tested the
    wrong thing."""
    with pytest.raises(KeyError, match="unknown config keys"):
        config_from_dict({"train": {"total_stps": 100}})


def test_unknown_section_is_rejected():
    with pytest.raises(KeyError, match="unknown top-level"):
        config_from_dict({"nonsense": {}})


def test_nested_dataclasses_are_built_from_mappings():
    cfg = config_from_dict({
        "agent": {"lr": 0.001, "net": {"hidden_sizes": [32], "num_heads": 3},
                  "replay": {"capacity": 500}}})
    assert cfg.agent.lr == 0.001
    assert cfg.agent.net.hidden_sizes == (32,)   # list coerced to tuple
    assert cfg.agent.net.num_heads == 3
    assert cfg.agent.replay.capacity == 500
    assert cfg.agent.gamma == 0.99               # untouched default


def test_all_shipped_configs_load():
    """Every policy/env/experiment config must actually parse.

    Cheap insurance: a broken config would otherwise only surface partway
    through a multi-hour sweep.
    """
    import itertools
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in itertools.chain(
        (root / "env").glob("*.yaml"),
        (root / "policy").glob("*.yaml"),
        (root / "experiments").glob("*.yaml"),
    ):
        rel = path.relative_to(root).with_suffix("").as_posix()
        data = load_yaml_tree(rel)
        assert isinstance(data, dict), rel


def test_run_id_is_stable_and_directory_safe():
    cfg = load_config("experiments/smoke", ["run.seed=3"])
    assert cfg.run_id == "CartPole-v1/eps_greedy/seed3"
