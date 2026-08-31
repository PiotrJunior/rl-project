"""Q-networks: dueling / plain heads, optionally an ensemble of heads.

One class covers every architecture the study needs, because the ensemble arm
must be architecturally identical to its control arm except for the head count.
Building them from the same code removes a whole class of "did I accidentally
change two things at once" confound.

Shapes: ``forward`` always returns ``(N, num_heads, num_actions)``.  Callers
that do not care about heads use :meth:`q_values`, which averages them.  Keeping
the head axis present even when ``num_heads == 1`` means the agent and the
uncertainty estimators need no special-casing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from e2b.nets.torso import MlpTorso, NatureCnnTorso


class QHead(nn.Module):
    """A single Q-head, dueling or plain."""

    def __init__(self, in_dim: int, num_actions: int, dueling: bool = True) -> None:
        super().__init__()
        self.dueling = dueling
        self.num_actions = num_actions
        if dueling:
            self.advantage = nn.Linear(in_dim, num_actions)
            self.value = nn.Linear(in_dim, 1)
        else:
            self.q = nn.Linear(in_dim, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.q(features)
        advantage = self.advantage(features)
        value = self.value(features)
        # Mean-centred advantage (Wang et al. 2016): identifiable and more
        # stable in practice than the max-centred form.
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """Torso + one or more Q-heads."""

    def __init__(
        self,
        torso: nn.Module,
        num_actions: int,
        dueling: bool = True,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.torso = torso
        self.num_actions = num_actions
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [QHead(torso.out_dim, num_actions, dueling) for _ in range(num_heads)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns ``(N, num_heads, num_actions)``."""
        features = self.torso(obs)
        return torch.stack([head(features) for head in self.heads], dim=1)

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean Q over heads: ``(N, num_actions)``.

        This is the Q the exploration policy sees.  Averaging (rather than
        picking one head per episode, as Bootstrapped DQN does for its own
        exploration) is deliberate: here the ensemble is a *measurement device*
        for uncertainty, and the exploration is supplied by the policy under
        study.  Letting the ensemble also drive exploration would confound the
        two.
        """
        return self.forward(obs).mean(dim=1)


def build_qnetwork(
    obs_shape: tuple[int, ...],
    num_actions: int,
    hidden_sizes: Sequence[int] = (128, 128),
    dueling: bool = True,
    num_heads: int = 1,
    kind: str = "mlp",
    cnn_channels: Sequence[int] = (32, 64, 64),
    cnn_kernels: Sequence[int] = (8, 4, 3),
    cnn_strides: Sequence[int] = (4, 2, 1),
    cnn_hidden: int = 512,
) -> QNetwork:
    if kind == "mlp":
        if len(obs_shape) != 1:
            raise ValueError(f"mlp torso needs a 1-D observation, got {obs_shape}")
        torso: nn.Module = MlpTorso(obs_shape[0], hidden_sizes)
    elif kind == "cnn":
        if len(obs_shape) != 3:
            raise ValueError(f"cnn torso needs a (C, H, W) observation, got {obs_shape}")
        torso = NatureCnnTorso(
            in_channels=obs_shape[0],
            screen_size=obs_shape[1],
            channels=cnn_channels,
            kernels=cnn_kernels,
            strides=cnn_strides,
            hidden=cnn_hidden,
        )
    else:
        raise KeyError(f"unknown network kind {kind!r}; expected 'mlp' or 'cnn'")
    return QNetwork(torso, num_actions, dueling=dueling, num_heads=num_heads)
