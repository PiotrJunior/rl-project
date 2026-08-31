"""Construction of exploration policies from config.

Policy configs name a variant and supply its knobs::

    policy:
      name: eps_boltzmann
      epsilon:    {type: linear,     start: 1.0,   end: 0.01, duration: 150000}
      temperature:{type: log_linear, start: 1.0e-4, end: 0.3, duration: 150000}
      top_k: num_actions

The literal string ``num_actions`` is substituted with the environment's action
count wherever it appears, so a config can express "all actions" or "anneal k
from 1 to |A|" without hard-coding a number per environment.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from e2b.policies.base import ExplorationPolicy
from e2b.policies.unified import MixturePolicy, UnifiedPolicy
from e2b.policies.uncertainty_gated import UncertaintyGatedPolicy

NUM_ACTIONS_SENTINEL = "num_actions"


def _substitute(value: Any, num_actions: int, total_steps: int) -> Any:
    """Resolve the two config sentinels.

    * the literal string ``num_actions`` -> the environment's action count;
    * ``duration_frac: f`` inside a schedule -> ``duration: round(f * total_steps)``.

    ``duration_frac`` is what lets a single policy config express "anneal over
    the first half of training" and be applied unchanged to CartPole (100k
    steps) and LunarLander (400k steps). Writing absolute durations instead
    would silently give the same named variant a different schedule on every
    environment, which would make the cross-environment comparison meaningless.
    """
    if isinstance(value, str) and value == NUM_ACTIONS_SENTINEL:
        return num_actions
    if isinstance(value, Mapping):
        out = {
            k: _substitute(v, num_actions, total_steps)
            for k, v in value.items()
            if k not in ("duration_frac", "delay_frac")
        }
        if "duration_frac" in value:
            if "duration" in value:
                raise KeyError("specify either `duration` or `duration_frac`, not both")
            out["duration"] = int(round(float(value["duration_frac"]) * total_steps))
        if "delay_frac" in value:
            if "delay" in value:
                raise KeyError("specify either `delay` or `delay_frac`, not both")
            out["delay"] = int(round(float(value["delay_frac"]) * total_steps))
        return out
    if isinstance(value, list):
        return [_substitute(v, num_actions, total_steps) for v in value]
    return value


def _unified(num_actions: int, **kwargs: Any) -> UnifiedPolicy:
    return UnifiedPolicy(num_actions=num_actions, **kwargs)


def _mixture(num_actions: int, **kwargs: Any) -> MixturePolicy:
    """Idea 1a: mix a full epsilon-greedy policy with a full Boltzmann policy.

    The two components are themselves ``UnifiedPolicy`` instances, so the
    mixture endpoints are the *same objects* the dedicated baselines use --
    there is no second implementation of either endpoint that could drift.
    """
    kwargs = dict(kwargs)
    beta = kwargs.pop("beta", 0.0)
    first_cfg = dict(kwargs.pop("first", {"epsilon": 0.05, "top_k": 1}))
    second_cfg = dict(kwargs.pop("second", {"epsilon": 0.0, "temperature": 0.3}))
    if kwargs:
        raise KeyError(f"unexpected keys for mixture policy: {sorted(kwargs)}")
    first = UnifiedPolicy(num_actions=num_actions, **first_cfg)
    second = UnifiedPolicy(num_actions=num_actions, **second_cfg)
    return MixturePolicy(num_actions, first=first, second=second, beta=beta)


def _uncertainty_gated(num_actions: int, **kwargs: Any) -> UncertaintyGatedPolicy:
    return UncertaintyGatedPolicy(num_actions=num_actions, **kwargs)


# Each entry is a builder; the *variants* of the study are distinguished by the
# knob values in configs/policy/*.yaml, not by separate classes. That is the
# point of the unified family -- see e2b/policies/unified.py.
_BUILDERS = {
    "eps_greedy": _unified,
    "boltzmann": _unified,
    "eps_boltzmann": _unified,
    "topk_boltzmann": _unified,
    "anneal_k": _unified,
    "topp_boltzmann": _unified,
    "unified": _unified,
    "mixture_anneal": _mixture,
    "uncertainty_gated": _uncertainty_gated,
}


def available_policies() -> list[str]:
    return sorted(_BUILDERS)


def build_policy(
    cfg: Mapping[str, Any], num_actions: int, total_steps: int = 0
) -> ExplorationPolicy:
    """Instantiate the exploration policy described by ``cfg``."""
    cfg = copy.deepcopy(dict(cfg))
    name = cfg.pop("name", None)
    if name is None:
        raise KeyError("policy config must have a `name`")
    if name not in _BUILDERS:
        raise KeyError(
            f"unknown policy {name!r}; available: {available_policies()}"
        )
    cfg = _substitute(cfg, num_actions, total_steps)
    policy = _BUILDERS[name](num_actions, **cfg)
    policy.name = name
    return policy


def requires_uncertainty(cfg: Mapping[str, Any]) -> bool:
    """Whether this policy needs an uncertainty signal on the acting path."""
    return cfg.get("name") == "uncertainty_gated"
