from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVideoEncoder(nn.Module):
    """Lightweight 3D CNN encoder for video clips without temporal downsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        emb_dim: int = 256,
        use_time_film: bool = False,
    ) -> None:
        super().__init__()
        self.use_time_film = use_time_film

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv4 = nn.Conv3d(128, hidden_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.norm1 = nn.BatchNorm3d(32)
        self.norm2 = nn.BatchNorm3d(64)
        self.norm3 = nn.BatchNorm3d(128)
        self.norm4 = nn.BatchNorm3d(hidden_dim)

        # Keep time dimension, pool spatially, and project to embeddings.
        self.proj = nn.Conv3d(hidden_dim, emb_dim, kernel_size=1)

        if use_time_film:
            self.time_film = nn.Sequential(
                nn.Conv1d(emb_dim, emb_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(emb_dim, emb_dim * 2, kernel_size=1),
            )
        else:
            self.time_film = None

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        x = F.silu(self.norm4(self.conv4(x)))

        # Global spatial pooling, keep time steps.
        x = x.mean(dim=[3, 4], keepdim=True)  # (B, hidden_dim, T, 1, 1)
        x = self.proj(x).squeeze(-1).squeeze(-1)  # (B, emb_dim, T)

        if self.time_film is not None:
            gamma_beta = self.time_film(x)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            gamma = 0.1 * torch.tanh(gamma)
            x = x * (1 + gamma) + beta

        return x.permute(0, 2, 1)  # (B, T, emb_dim)


__all__ = ["SimpleVideoEncoder"]
