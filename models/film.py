from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation using a conditioning embedding."""

    def __init__(self, in_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)
        gamma_beta = self.linear(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
