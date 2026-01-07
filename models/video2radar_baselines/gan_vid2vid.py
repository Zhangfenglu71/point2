"""Vid2Vid-GAN (2018) baseline: video-conditioned generator + patch discriminator.

Key difference: a single GAN generator internally encodes the video clip and upsamples
into a spectrogram, while the discriminator is a lightweight 2D patch classifier.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.film import FiLM
from models.video_encoder import SimpleVideoEncoder


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.film = FiLM(out_channels, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.film(x, cond)
        return x


class Vid2VidGANGenerator(nn.Module):
    def __init__(
        self,
        img_size: int = 120,
        radar_channels: int = 1,
        base_channels: int = 64,
        z_dim: int = 128,
        cond_dim: int = 256,
        video_encoder_type: str = "temporal_unet",
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.radar_channels = radar_channels
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.video_encoder = SimpleVideoEncoder(emb_dim=cond_dim, encoder_type=video_encoder_type)
        start_size = max(4, img_size // 16)
        self.start_size = start_size
        self.init_proj = nn.Linear(cond_dim + z_dim, base_channels * start_size * start_size)
        self.blocks = nn.ModuleList()
        channels = base_channels
        cur_size = start_size
        while cur_size < img_size:
            next_channels = max(16, channels // 2)
            self.blocks.append(_UpsampleBlock(channels, next_channels, cond_dim))
            channels = next_channels
            cur_size *= 2
        self.to_rgb = nn.Conv2d(channels, radar_channels, kernel_size=3, padding=1)

    def forward(
        self,
        video: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(video.size(0), self.z_dim, device=video.device)
        cond = self.video_encoder(video, labels)
        cond_vec = cond.vector
        latent = torch.cat([cond_vec, noise], dim=1)
        x = self.init_proj(latent)
        x = x.view(video.size(0), -1, self.start_size, self.start_size)
        for block in self.blocks:
            x = block(x, cond_vec)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = torch.tanh(self.to_rgb(x))
        return x


class Vid2VidGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
