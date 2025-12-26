from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVideoEncoder(nn.Module):
    """Lightweight 3D CNN encoder for video clips."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, emb_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm3d(32)
        self.norm2 = nn.BatchNorm3d(64)
        self.norm3 = nn.BatchNorm3d(128)
        self.norm4 = nn.BatchNorm3d(hidden_dim)
        self.proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = x.mean(dim=[2, 3, 4])
        emb = self.proj(x)
        return emb
