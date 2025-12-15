"""Simple MLP displacement network for strong-form PINNs."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Fourier feature mapping to help with high-frequency content."""

    def __init__(self, in_dim: int, num_frequencies: int = 6, sigma: float = 1.0):
        super().__init__()
        B = torch.randn((num_frequencies, in_dim)) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim)
        proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DisplacementMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        hidden_layers: Iterable[int] | None = None,
        activation: nn.Module | None = None,
        use_fourier: bool = False,
        fourier_frequencies: int = 6,
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers) if hidden_layers is not None else [128] * 8
        activation = activation or nn.Tanh()

        self.use_fourier = use_fourier
        if use_fourier:
            self.ff = FourierFeatures(in_dim, num_frequencies=fourier_frequencies)
            first_in = fourier_frequencies * 2
        else:
            first_in = in_dim

        layers: List[nn.Module] = []
        prev = first_in
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.ff(x) if self.use_fourier else x
        return self.net(features)
