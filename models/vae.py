from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramVAE(nn.Module):
    """Simple 2D convolutional VAE for color radar spectrograms."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        num_downsamples: int = 2,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.num_downsamples = num_downsamples
        enc_layers = []
        ch = in_channels
        for i in range(num_downsamples):
            out_ch = base_channels * (2**i)
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, stride=1),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                )
            )
            ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)
        self.to_mu = nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(ch, latent_channels, kernel_size=3, padding=1)

        dec_layers = []
        ch = latent_channels
        for i in reversed(range(num_downsamples)):
            out_ch = base_channels * (2**i)
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                )
            )
            ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        self.final_conv = nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)

    @property
    def downsample_factor(self) -> int:
        return 2**self.num_downsamples

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        x_recon = torch.tanh(self.final_conv(h))
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_l1 = F.l1_loss(recon, x)
        loss = recon_l1 + self.beta * kl
        return recon, mu, logvar, loss
