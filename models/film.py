from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation using a conditioning embedding."""

    def __init__(self, in_channels: int, cond_dim: int, gamma_scale: float = 0.1) -> None:
        super().__init__()
        self.gamma_scale = float(gamma_scale)
        self.linear = nn.Linear(cond_dim, in_channels * 2)

        # Stabilize early training: small weights and zero bias keep modulation near identity.
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)
        cond_fp32 = cond.float()
        gamma_beta = self.linear(cond_fp32).to(x.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        # Limit modulation magnitude to avoid exploding/negative scales.
        gamma = self.gamma_scale * torch.tanh(gamma)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
