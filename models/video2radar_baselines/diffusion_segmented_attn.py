"""Segmented Cross-Attn VideoDiffusion baseline.

Key difference: video tokens are split into temporal segments and each U-Net scale
attends to its assigned segment token only, enforcing segmented cross-attention.
"""

from __future__ import annotations

import torch

from models.conditioning import UNetConditioning
from models.video2radar_baselines.diffusion_common import VideoDiffusionUNetBase


class VideoDiffusionSegmentedAttn(VideoDiffusionUNetBase):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        cond_dim: int = 256,
        video_encoder_type: str = "temporal_unet",
        cross_heads: int = 4,
        num_segments: int = 4,
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
        self.num_segments = num_segments

    def process_conditioning(self, cond: UNetConditioning, t: torch.Tensor) -> UNetConditioning:
        if not cond.scale_tokens:
            return cond
        tokens = cond.scale_tokens[0]
        b, t_len, dim = tokens.shape
        seg_size = max(1, t_len // self.num_segments)
        segment_tokens = []
        for seg_idx in range(self.num_segments):
            start = seg_idx * seg_size
            end = t_len if seg_idx == self.num_segments - 1 else min(t_len, (seg_idx + 1) * seg_size)
            seg = tokens[:, start:end, :]
            if seg.numel() == 0:
                seg = tokens[:, :1, :]
            segment_tokens.append(seg.mean(dim=1))
        segment_tokens = torch.stack(segment_tokens, dim=1)
        new_tokens = []
        for idx in range(len(cond.scale_tokens)):
            seg_idx = idx % self.num_segments
            seg_tok = segment_tokens[:, seg_idx : seg_idx + 1, :]
            new_tokens.append(seg_tok)
        return UNetConditioning(vector=cond.vector, scale_tokens=new_tokens)
