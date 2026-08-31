"""Deterministic seeding across python, numpy and torch.

Reproducibility matters more than usual here: the effects we are trying to
measure (an exploration schedule shifting a learning curve) are small relative
to DQN's seed variance, so a run must be exactly repeatable before any claim
about a difference between variants is meaningful.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG this project touches.

    ``deterministic_torch`` trades a little speed for bitwise reproducibility.
    On the small MLPs used for the Gym experiments the cost is negligible, so it
    defaults on.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """A dedicated numpy Generator.

    Exploration policies take their own Generator rather than using the global
    numpy RNG, so that changing the *number* of network updates (which consumes
    torch RNG) does not shift the action-sampling stream. Without this, two
    variants that should be identical in the limit are not bitwise identical,
    and the limiting-case tests become flaky.
    """
    return np.random.default_rng(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a torch device.

    ``auto`` deliberately prefers CPU for this project.  The Gym experiments use
    two-layer MLPs with batch size 64, where per-kernel dispatch overhead
    dominates and MPS/CUDA are measurably *slower* than CPU while also
    serialising the parallel sweep onto one accelerator.  Pass ``mps``/``cuda``
    explicitly for the (convolutional) Atari code path, where it does pay off.
    """
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cpu")
