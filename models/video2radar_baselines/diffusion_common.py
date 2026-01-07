"""Shared utilities for video-conditioned diffusion baselines."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.conditioning import UNetConditioning
from models.unet import UNet
from models.video_encoder import SimpleVideoEncoder


class VideoDiffusionUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
        video_encoder_type: str = "temporal_unet",
        use_cross_attn: bool = True,
        cross_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.video_encoder = SimpleVideoEncoder(emb_dim=cond_dim, encoder_type=video_encoder_type)
        self.unet = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            cond_dim=cond_dim,
            use_film=False,
            use_cross_attn=use_cross_attn,
            cross_heads=cross_heads,
            channel_mults=channel_mults,
        )

    def process_conditioning(self, cond: UNetConditioning, t: torch.Tensor) -> UNetConditioning:
        return cond

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        video: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.video_encoder(video, labels)
        cond = self.process_conditioning(cond, t)
        return self.unet(x, t, cond)
