"""3DUNet-VideoDiffusion baseline.

Key difference: a single 3D U-Net consumes the video clip plus noisy spectrogram (tiled
along time) and predicts denoising residuals before a 2D head pools over time. The
implementation supports optional label conditioning by injecting class embeddings into
the time embedding stream, matching the conditioning used by other video baselines.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (half_dim - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class _ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1, 1)
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)


class VideoDiffusion3DUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        video_channels: int = 3,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.in_channels = in_channels
        self.video_channels = video_channels
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.label_embed = nn.Embedding(num_classes, time_dim)
        enc_blocks = []
        downs = []
        ch = in_channels + video_channels
        channels = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            enc_blocks.append(_ResidualBlock3D(ch, out_ch, time_dim))
            channels.append(out_ch)
            ch = out_ch
        for c in channels[:-1]:
            downs.append(nn.Conv3d(c, c, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downsamples = nn.ModuleList(downs)
        self.mid_block1 = _ResidualBlock3D(ch, ch, time_dim)
        self.mid_block2 = _ResidualBlock3D(ch, ch, time_dim)
        dec_blocks = []
        ups = []
        skip_channels = list(reversed(channels[1:]))
        for mult, skip_ch in zip(reversed(channel_mults[:-1]), skip_channels):
            out_ch = base_channels * mult
            dec_blocks.append(_ResidualBlock3D(ch + skip_ch, out_ch, time_dim))
            ups.append(nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)))
            ch = out_ch
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.upsamples = nn.ModuleList(ups)
        self.final_block = _ResidualBlock3D(ch + in_channels + video_channels, base_channels, time_dim)
        self.out_conv = nn.Conv3d(base_channels, in_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, video: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, C, H, W), video: (B, T, 3, H, W)
        b, _, h, w = x.shape
        t_emb = _sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        if labels is not None:
            t_emb = t_emb + self.label_embed(labels)
        video = video.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        x_rep = x.unsqueeze(2).repeat(1, 1, video.size(2), 1, 1)
        h3d = torch.cat([x_rep, video], dim=1)
        hs = []
        for idx, block in enumerate(self.enc_blocks):
            h3d = block(h3d, t_emb)
            hs.append(h3d)
            if idx < len(self.downsamples):
                h3d = self.downsamples[idx](h3d)
        h3d = self.mid_block1(h3d, t_emb)
        h3d = self.mid_block2(h3d, t_emb)
        skips = list(reversed(hs[:-1]))
        for idx, block in enumerate(self.dec_blocks):
            skip = skips[idx]
            if h3d.shape[2:] != skip.shape[2:]:
                h3d = F.interpolate(h3d, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h3d = torch.cat([h3d, skip], dim=1)
            h3d = block(h3d, t_emb)
            h3d = self.upsamples[idx](h3d)
        if h3d.shape[2:] != (video.size(2), h, w):
            h3d = F.interpolate(h3d, size=(video.size(2), h, w), mode="trilinear", align_corners=False)
        h3d = torch.cat([h3d, x_rep, video], dim=1)
        h3d = self.final_block(h3d, t_emb)
        out = self.out_conv(h3d)
        out = out.mean(dim=2)
        return out
