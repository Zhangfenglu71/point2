"""Cross-Attn-Control VideoDiffusion baseline.

Key difference: cross-attention tokens are gated by a learned control signal derived
from the time embedding and video conditioning vector, enforcing attention control
without external networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.conditioning import UNetConditioning
from models.unet import sinusoidal_time_embedding
from models.video2radar_baselines.diffusion_common import VideoDiffusionUNetBase


class _AttentionGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, cond_vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, cond_vec.size(1))
        gate = torch.sigmoid(self.mlp(cond_vec + t_emb))
        return gate


class VideoDiffusionAttnControl(VideoDiffusionUNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
        video_encoder_type: str = "temporal_unet",
        cross_heads: int = 4,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            cond_dim=cond_dim,
            video_encoder_type=video_encoder_type,
            use_cross_attn=True,
            cross_heads=cross_heads,
        )
        self.gate = _AttentionGate(cond_dim)

    def process_conditioning(self, cond: UNetConditioning, t: torch.Tensor) -> UNetConditioning:
        if cond.vector is None:
            return cond
        gate = self.gate(cond.vector, t)
        gated_vec = cond.vector * gate
        if cond.scale_tokens is None:
            return UNetConditioning(vector=gated_vec, scale_tokens=None)
        gated_tokens = [tok * gate.unsqueeze(1) for tok in cond.scale_tokens]
        return UNetConditioning(vector=gated_vec, scale_tokens=gated_tokens)
