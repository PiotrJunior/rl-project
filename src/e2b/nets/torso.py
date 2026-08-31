"""Feature extractors shared by every Q-head variant."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MlpTorso(nn.Module):
    """Two-layer (by default) MLP for low-dimensional state vectors."""

    def __init__(self, in_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last, size))
            layers.append(nn.ReLU())
            last = size
        self.net = nn.Sequential(*layers)
        self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NatureCnnTorso(nn.Module):
    """The DQN-Nature convolutional torso, used by the Atari code path.

    Input is expected as uint8 ``(N, C, H, W)`` with ``C`` = frame-stack depth;
    scaling to [0, 1] happens here so the replay buffer can store uint8 frames
    and use ~4x less memory than float32 would.
    """

    def __init__(
        self,
        in_channels: int,
        screen_size: int,
        channels: Sequence[int] = (32, 64, 64),
        kernels: Sequence[int] = (8, 4, 3),
        strides: Sequence[int] = (4, 2, 1),
        hidden: int = 512,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("channels, kernels and strides must have equal length")
        conv: list[nn.Module] = []
        last = in_channels
        for c, k, s in zip(channels, kernels, strides):
            conv.append(nn.Conv2d(last, c, kernel_size=k, stride=s))
            conv.append(nn.ReLU())
            last = c
        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, screen_size, screen_size)
            n_flat = self.conv(dummy).flatten(1).shape[1]

        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, hidden), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        return self.head(self.conv(x))
