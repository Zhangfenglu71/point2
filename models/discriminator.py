from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.real_video_radar import ACTIONS


class ACDiscriminator(nn.Module):
    """Simple AC-GAN style discriminator for radar spectrograms."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(base_channels * 4)
        self.head_adv = nn.Linear(base_channels * 4, 1)
        self.head_cls = nn.Linear(base_channels * 4, len(ACTIONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2, inplace=True)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2, inplace=True)
        h = F.leaky_relu(self.bn4(self.conv4(h)), 0.2, inplace=True)
        h = h.mean(dim=(2, 3))
        adv = self.head_adv(h)
        cls = self.head_cls(h)
        return adv, cls


__all__ = ["ACDiscriminator"]
