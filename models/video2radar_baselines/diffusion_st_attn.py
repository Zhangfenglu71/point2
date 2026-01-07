"""ST-Attn-VideoDiffusion baseline.

Key difference: temporal self-attention refines video tokens before they are injected as
cross-attention keys/values within the diffusion U-Net.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.conditioning import UNetConditioning
from models.video2radar_baselines.diffusion_common import VideoDiffusionUNetBase


class _TemporalTokenAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        return tokens + attn_out


class VideoDiffusionSTAttn(VideoDiffusionUNetBase):
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
        self.token_attn = nn.ModuleList([
            _TemporalTokenAttn(cond_dim, max(1, cross_heads)) for _ in range(len(channel_mults))
        ])

    def process_conditioning(self, cond: UNetConditioning, t: torch.Tensor) -> UNetConditioning:
        if not cond.scale_tokens:
            return cond
        new_tokens = []
        for idx, token in enumerate(cond.scale_tokens):
            attn = self.token_attn[min(idx, len(self.token_attn) - 1)]
            new_tokens.append(attn(token))
        return UNetConditioning(vector=cond.vector, scale_tokens=new_tokens)
