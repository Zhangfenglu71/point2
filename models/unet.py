from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning import UNetConditioning
from models.film import FiLM


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        cond_dim: Optional[int] = None,
        use_film: bool = False,
        use_cross_attn: bool = False,
        cross_heads: int = 4,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        self.cond_dim = cond_dim
        self.use_cross_attn = use_cross_attn and cond_dim is not None
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        if self.use_cross_attn:
            self.cross_q_proj = nn.Linear(out_channels, out_channels)
            self.cross_k_proj = nn.Linear(cond_dim, out_channels)
            self.cross_v_proj = nn.Linear(cond_dim, out_channels)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=out_channels, num_heads=cross_heads, batch_first=True
            )
            self.cross_out = nn.Linear(out_channels, out_channels)
        if use_film and cond_dim is not None:
            self.film = FiLM(out_channels, cond_dim)
        else:
            self.film = None

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        cond_vec: Optional[torch.Tensor],
        cross_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        time_term = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term
        if self.use_cross_attn:
            cond_tokens = cross_tokens
            if cond_tokens is None and cond_vec is not None:
                cond_tokens = cond_vec.unsqueeze(1)
            if cond_tokens is not None:
                B, C, H, W = h.shape
                tokens = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
                tokens = self.cross_q_proj(tokens)
                cond_fp32 = cond_tokens.float()
                k = self.cross_k_proj(cond_fp32)
                v = self.cross_v_proj(cond_fp32)
                attn_out, _ = self.cross_attn(tokens, k, v)
                attn_out = self.cross_out(attn_out).to(h.dtype)
                attn_out = attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
                h = h + attn_out
        if self.film is not None and cond_vec is not None:
            h = self.film(h, cond_vec)
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        cond_dim: Optional[int] = None,
        use_film: bool = False,
        use_cross_attn: bool = False,
        cross_heads: int = 4,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.use_cond = cond_dim is not None
        self.cond_dim = cond_dim
        self.use_film = use_film and cond_dim is not None
        self.use_cross_attn = use_cross_attn and cond_dim is not None
        self.cross_heads = cross_heads
        self.channel_mults = channel_mults

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Encoder
        enc_blocks = []
        in_ch = in_channels
        channels = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            enc_blocks.append(
                ResidualBlock(
                    in_ch,
                    out_ch,
                    time_dim,
                    cond_dim,
                    self.use_film,
                    self.use_cross_attn,
                    cross_heads,
                )
            )
            channels.append(out_ch)
            in_ch = out_ch
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downsamples = nn.ModuleList(
            [nn.Conv2d(c, c, 4, stride=2, padding=1) for c in channels[:-1]]
        )

        # Bottleneck
        self.mid_block1 = ResidualBlock(
            in_ch, in_ch, time_dim, cond_dim, self.use_film, self.use_cross_attn, cross_heads
        )
        self.mid_block2 = ResidualBlock(
            in_ch, in_ch, time_dim, cond_dim, self.use_film, self.use_cross_attn, cross_heads
        )

        # Decoder
        dec_blocks = []
        upsamples = []
        self.dec_in_channels = []
        self.decoder_projections = nn.ModuleList()
        skip_channels = list(reversed(channels[1:]))
        decoder_mults = list(reversed(channel_mults[:-1]))
        for mult, skip_ch in zip(decoder_mults, skip_channels):
            out_ch = base_channels * mult
            dec_in = in_ch + skip_ch
            self.dec_in_channels.append(dec_in)
            self.decoder_projections.append(nn.Identity())
            dec_blocks.append(
                ResidualBlock(
                    dec_in,
                    out_ch,
                    time_dim,
                    cond_dim,
                    self.use_film,
                    self.use_cross_attn,
                    cross_heads,
                )
            )
            upsamples.append(nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.upsamples = nn.ModuleList(upsamples)

        self.final_block = ResidualBlock(
            in_ch + in_channels,
            base_channels,
            time_dim,
            cond_dim,
            self.use_film,
            self.use_cross_attn,
            cross_heads,
        )
        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

        # If conditioning without FiLM, project cond to time_dim and add to time embedding.
        if self.use_cond and not self.use_film:
            self.cond_to_time = nn.Linear(cond_dim, time_dim)
        else:
            self.cond_to_time = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[UNetConditioning | torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W); t: (B,)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        if isinstance(cond, UNetConditioning):
            cond_obj = cond
        elif cond is None:
            cond_obj = None
        else:
            cond_obj = UNetConditioning(vector=cond)
        cond_vec = cond_obj.vector if cond_obj is not None else None
        cond_tokens = cond_obj.scale_tokens if cond_obj is not None else None
        if self.use_cond and not self.use_film and cond_vec is not None:
            t_emb = t_emb + self.cond_to_time(cond_vec)

        # Encoder
        hs = []
        h = x
        for i, block in enumerate(self.enc_blocks):
            token = cond_tokens[i] if cond_tokens is not None and i < len(cond_tokens) else None
            h = block(h, t_emb, cond_vec, token)
            hs.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Middle
        mid_token = cond_tokens[-1] if cond_tokens else None
        h = self.mid_block1(h, t_emb, cond_vec, mid_token)
        h = self.mid_block2(h, t_emb, cond_vec, mid_token)
        mid_feat = torch.mean(h, dim=(2, 3))

        # Decoder
        for i, block in enumerate(self.dec_blocks):
            skip = hs[-(i + 1)]
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            expected_in = self.dec_in_channels[i]
            if h.shape[1] != expected_in:
                adapter = self.decoder_projections[i]
                if isinstance(adapter, nn.Identity) or getattr(adapter, "in_channels", None) != h.shape[1]:
                    adapter = nn.Conv2d(h.shape[1], expected_in, 1)
                    self.decoder_projections[i] = adapter.to(h.device)
                h = adapter(h)
            token_idx = (len(cond_tokens) - 1 - i) if cond_tokens else None
            token = cond_tokens[token_idx] if token_idx is not None and token_idx >= 0 else None
            h = block(h, t_emb, cond_vec, token)
            h = self.upsamples[i](h)

        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, x], dim=1)
        final_token = cond_tokens[0] if cond_tokens else None
        h = self.final_block(h, t_emb, cond_vec, final_token)
        out = self.out_conv(h)
        if return_features:
            return out, mid_feat
        return out
