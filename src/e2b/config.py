"""Configuration: typed dataclasses + composable YAML.

A run is fully described by one resolved config, which is dumped into the run
directory.  Configs compose through a ``defaults`` list, so an experiment file
is only the *difference* from the base:

    # configs/experiments/main_gym.yaml
    defaults: [base, env/lunarlander, policy/eps_boltzmann]
    train: {total_steps: 400000}

This is what keeps the study honest -- every arm inherits the identical
``agent``/``train`` block and differs only in ``policy``, so a measured
difference cannot be attributed to a stray hyperparameter.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class EnvConfig:
    id: str = "CartPole-v1"
    kind: str = "gym"  # "gym" | "atari"
    max_episode_steps: int | None = None
    # Atari-only knobs; ignored for kind == "gym".
    frame_skip: int = 4
    frame_stack: int = 4
    screen_size: int = 84
    grayscale: bool = True
    terminal_on_life_loss: bool = False
    repeat_action_probability: float = 0.25
    clip_rewards: bool = True
    noop_max: int = 30


@dataclass
class NetConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    dueling: bool = True
    # Ensemble is only used by the uncertainty-gated extension. num_heads == 1
    # means a plain single-head network.
    num_heads: int = 1
    bootstrap_prob: float = 0.8
    # Atari CNN
    cnn_channels: tuple[int, ...] = (32, 64, 64)
    cnn_kernels: tuple[int, ...] = (8, 4, 3)
    cnn_strides: tuple[int, ...] = (4, 2, 1)
    cnn_hidden: int = 512


@dataclass
class ReplayConfig:
    capacity: int = 100_000
    prioritized: bool = True
    alpha: float = 0.5
    beta_start: float = 0.4
    beta_end: float = 1.0
    priority_eps: float = 1e-6
    n_step: int = 3


@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    batch_size: int = 64
    double_dqn: bool = True
    target_update_interval: int = 1_000
    train_frequency: int = 4
    learning_starts: int = 5_000
    max_grad_norm: float = 10.0
    huber_kappa: float = 1.0
    net: NetConfig = field(default_factory=NetConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    eval_interval: int = 10_000
    eval_episodes: int = 10
    diagnostics_interval: int = 2_000
    progress: bool = False


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "auto"
    out_dir: str = "results/runs"
    name: str = "run"
    torch_threads: int = 1


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)
    # Policy config stays an untyped mapping: each exploration variant has a
    # different set of knobs, and the registry validates them on construction.
    # Forcing them into one dataclass would mean a union of every variant's
    # fields, where nothing is required and typos pass silently.
    policy: dict[str, Any] = field(default_factory=lambda: {"name": "eps_greedy"})

    @property
    def run_id(self) -> str:
        """Directory-safe identifier: <name>/<env>/<policy>/seed<k>."""
        policy_name = self.policy.get("name", "unknown")
        return f"{self.env.id}/{policy_name}/seed{self.run.seed}"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    The ``policy`` block is merged as a whole rather than key-wise when the
    override changes ``policy.name``: two variants have disjoint knobs, so
    inheriting leftovers from the previous policy would silently mis-configure
    the new one.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key == "policy"
            and isinstance(value, Mapping)
            and "name" in value
            and isinstance(out.get(key), Mapping)
            and out[key].get("name") not in (None, value["name"])
        ):
            out[key] = copy.deepcopy(dict(value))
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_path(name: str, root: Path) -> Path:
    """Map a ``defaults`` entry such as ``env/cartpole`` to a YAML file."""
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml"):
        paths = [candidate, root / candidate]
    else:
        paths = [
            root / f"{name}.yaml",
            root / f"{name}.yml",
            Path(f"{name}.yaml"),
            Path(f"{name}.yml"),
        ]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"config {name!r} not found (looked in {[str(p) for p in paths]})"
    )


def load_yaml_tree(
    name: str, root: Path | None = None, _seen: set[Path] | None = None
) -> dict[str, Any]:
    """Load a YAML config, recursively resolving its ``defaults`` list.

    Earlier entries in ``defaults`` are overridden by later ones, and the file's
    own keys override all of them -- the same precedence Hydra uses, so it reads
    the way people expect.
    """
    root = root or CONFIG_ROOT
    path = _resolve_path(name, root)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config include involving {path}")
    _seen = _seen | {path}

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config {path} must contain a mapping at the top level")

    defaults = raw.pop("defaults", []) or []
    merged: dict[str, Any] = {}
    for parent in defaults:
        merged = _deep_merge(merged, load_yaml_tree(parent, root, _seen))
    return _deep_merge(merged, raw)


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce YAML scalars into the dataclass field type.

    Mainly this turns lists into the tuples used for layer sizes, and honours
    ``X | None`` fields.
    """
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _from_mapping(cls: type, data: Mapping[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a mapping, rejecting unknown keys.

    Rejecting unknown keys is deliberate: a typo like ``temprature`` in an
    experiment config would otherwise be silently ignored and produce a run that
    looks fine but tested the wrong thing.

    Note this module uses ``from __future__ import annotations``, so
    ``Field.type`` is a *string*.  Types are resolved via ``get_type_hints``.
    """
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(
            f"unknown config keys for {cls.__name__}: {sorted(unknown)}; "
            f"expected some of {sorted(known)}"
        )
    for name in known:
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name, Any)
        if is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[name] = _from_mapping(hint, value)
        else:
            kwargs[name] = _coerce(value, hint)
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> Config:
    data = dict(data)
    policy = dict(data.pop("policy", {"name": "eps_greedy"}))
    cfg = Config(policy=policy)
    for section in ("env", "agent", "train", "run"):
        if section in data:
            section_data = data.pop(section)
            if section == "agent":
                section_data = dict(section_data)
                net = section_data.pop("net", None)
                replay = section_data.pop("replay", None)
                agent = _from_mapping(AgentConfig, section_data)
                if net is not None:
                    agent.net = _from_mapping(NetConfig, net)
                if replay is not None:
                    agent.replay = _from_mapping(ReplayConfig, replay)
                cfg.agent = agent
            else:
                cls = {"env": EnvConfig, "train": TrainConfig, "run": RunConfig}[section]
                setattr(cfg, section, _from_mapping(cls, section_data))
    if data:
        raise KeyError(f"unknown top-level config sections: {sorted(data)}")
    return cfg


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` CLI overrides, parsing values as YAML.

    YAML parsing means ``train.total_steps=1000`` gives an int, ``run.device=cpu``
    gives a str, and ``agent.net.hidden_sizes=[64,64]`` gives a list, without
    any bespoke parsing rules.
    """
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be of the form key.path=value")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise TypeError(f"cannot descend into {key!r}: {part!r} is a scalar")
        cursor[parts[-1]] = value
    return out


def load_config(
    name: str, overrides: list[str] | None = None, root: Path | None = None
) -> Config:
    data = load_yaml_tree(name, root)
    if overrides:
        data = apply_overrides(data, overrides)
    return config_from_dict(data)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    out = asdict(cfg)
    # asdict turns tuples into lists already; keep policy as-is.
    out["policy"] = copy.deepcopy(cfg.policy)
    return out


def dump_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
