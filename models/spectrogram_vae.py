from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def _make_conv_block(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class SpectrogramVAE(nn.Module):
    """Compact convolutional VAE for colorful spectrograms.

    The encoder downsamples only spatial dimensions (no temporal striding) so the latent grid
    can be used by a 2D U-Net operating on frame sequences.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        beta: float = 0.1,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_dim
        self.beta = beta
        self.scaling_factor = scaling_factor

        hidden_dims = tuple(hidden_dims) if hidden_dims is not None else (64, 128, 256)

        enc_layers = []
        last_ch = in_channels
        for ch in hidden_dims:
            enc_layers.append(_make_conv_block(last_ch, ch))
            last_ch = ch
        self.encoder = nn.Sequential(*enc_layers)
        self.to_mu = nn.Conv2d(last_ch, latent_dim, kernel_size=3, padding=1)
        self.to_logvar = nn.Conv2d(last_ch, latent_dim, kernel_size=3, padding=1)

        dec_layers = []
        dec_channels = list(hidden_dims)[::-1]
        first_dec = dec_channels[0]
        self.decoder_input = nn.Conv2d(latent_dim, first_dec, kernel_size=3, padding=1)
        for idx in range(len(dec_channels) - 1):
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_channels[idx], dec_channels[idx + 1], kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, dec_channels[idx + 1]),
                    nn.SiLU(),
                )
            )
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(dec_channels[-1], dec_channels[-1], kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(dec_channels[-1], in_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*dec_layers)

    def _flatten_time(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            return x.view(b * t, c, h, w), t
        b, c, h, w = x.shape
        return x, 1

    def _unflatten_time(self, x: torch.Tensor, t: int) -> torch.Tensor:
        if t == 1:
            return x.unsqueeze(1)
        b_t, c, h, w = x.shape
        b = b_t // t
        return x.view(b, t, c, h, w)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat, t = self._flatten_time(x)
        h = self.encoder(flat)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z * self.scaling_factor
        return self._unflatten_time(z, t), self._unflatten_time(mu, t), self._unflatten_time(logvar, t)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        flat, t = self._flatten_time(z)
        z = flat / max(self.scaling_factor, 1e-6)
        h = self.decoder_input(z)
        recon = self.decoder(h)
        recon = self._unflatten_time(recon, t)
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        recon = self.decode_latents(z)
        return recon, mu, logvar, z

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * self.beta


__all__ = ["SpectrogramVAE"]
