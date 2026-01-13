from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioning import UNetConditioning
from models.film import FiLM
from models.unet import sinusoidal_time_embedding
from datasets.real_video_radar import ACTIONS


def _build_sinusoidal_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = _build_sinusoidal_embedding(max_len, dim, torch.device("cpu"))
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.pe.size(0):
            extra = _build_sinusoidal_embedding(length, self.pe.size(1), device)
            return extra
        return self.pe[:length].to(device)


class TemporalUNetEncoder(nn.Module):
    """Multi-scale 2D frame encoder with temporal FiLM + 1D Conv mixing (no temporal downsample)."""

    def __init__(
        self,
        in_channels: int,
        cond_dim: int,
        base_channels: int = 64,
        num_scales: int = 3,
        time_dim: int = 128,
        num_actions: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.num_actions = num_actions or len(ACTIONS)
        self.frame_blocks = nn.ModuleList()
        self.temporal_mixers = nn.ModuleList()
        self.token_projs = nn.ModuleList()
        self.token_norms = nn.ModuleList()
        self.films = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = in_channels
        for i in range(num_scales):
            out_ch = base_channels * (2**i)
            self.frame_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.GELU(),
                )
            )
            self.temporal_mixers.append(nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1))
            self.token_projs.append(nn.Linear(out_ch, cond_dim))
            self.token_norms.append(nn.LayerNorm(out_ch))
            self.films.append(FiLM(out_ch, cond_dim))
            if i < num_scales - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.time_proj = nn.Linear(time_dim, cond_dim)
        self.action_proj = nn.Linear(self.num_actions, cond_dim)
        self.global_norm = nn.LayerNorm(cond_dim)
        self.global_proj = nn.Linear(cond_dim, cond_dim)

    def forward(self, video: torch.Tensor, labels: Optional[torch.Tensor] = None) -> UNetConditioning:
        # video: (B, T, C, H, W)
        b, t, _, h, w = video.shape
        device = video.device
        positions = torch.arange(t, device=device)
        time_emb = sinusoidal_time_embedding(positions, self.time_dim)
        time_cond = self.time_proj(time_emb)  # (T, cond_dim)
        action_emb = None
        if labels is not None:
            one_hot = F.one_hot(labels, num_classes=self.num_actions).float()
            action_emb = self.action_proj(one_hot)

        scale_tokens: list[torch.Tensor] = []
        h_in = h
        w_in = w
        features = video
        for idx, frame_block in enumerate(self.frame_blocks):
            features = features.reshape(b * t, features.shape[2], h_in, w_in)
            features = frame_block(features)
            c_out = features.shape[1]
            features = features.reshape(b, t, c_out, h_in, w_in)
            cond = time_cond.unsqueeze(0).expand(b, t, -1)
            if action_emb is not None:
                cond = cond + action_emb.unsqueeze(1)
            features = self.films[idx](
                features.reshape(b * t, c_out, h_in, w_in), cond.reshape(b * t, -1)
            ).reshape(b, t, c_out, h_in, w_in)
            pooled = features.mean(dim=[3, 4])  # (B, T, C_out)
            temporal = self.temporal_mixers[idx](pooled.transpose(1, 2)).transpose(1, 2)
            token_feat = self.token_norms[idx](pooled + temporal)
            scale_tokens.append(self.token_projs[idx](token_feat))
            if idx < len(self.downsamples):
                features = self.downsamples[idx](features.reshape(b * t, c_out, h_in, w_in))
                h_in, w_in = features.shape[-2], features.shape[-1]
                features = features.reshape(b, t, c_out, h_in, w_in)

        global_token = scale_tokens[-1].mean(dim=1)
        if action_emb is not None:
            global_token = global_token + action_emb
        global_cond = self.global_proj(self.global_norm(global_token))
        return UNetConditioning(vector=global_cond, scale_tokens=scale_tokens)


class TimeSformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, t, hw, c = tokens.shape
        temporal_in = tokens.reshape(b * hw, t, c)
        temporal_norm = self.norm_t(temporal_in)
        temporal_out, _ = self.temporal_attn(temporal_norm, temporal_norm, temporal_norm)
        temporal_out = temporal_out + temporal_in
        temporal_out = temporal_out.reshape(b, hw, t, c).permute(0, 2, 1, 3)  # (B, T, HW, C)

        spatial_in = temporal_out.reshape(b * t, hw, c)
        spatial_norm = self.norm_s(spatial_in)
        spatial_out, _ = self.spatial_attn(spatial_norm, spatial_norm, spatial_norm)
        spatial_out = spatial_out + spatial_in
        spatial_out = spatial_out + self.mlp(self.norm_mlp(spatial_out))
        spatial_out = spatial_out.reshape(b, t, h * w, c)
        return spatial_out


class TimeSformerBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_heads: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            padding=(0, 0, 0),
        )
        self.blocks = nn.ModuleList([TimeSformerBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, C, H, W)
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.proj(x)
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, c)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        tokens = self.norm(tokens)
        return tokens.mean(dim=2)  # (B, T, C)


class ViT3DBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_heads: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            padding=(0, 0, 0),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.proj(x)
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        tokens = tokens.reshape(b, t, h * w, c)
        return tokens.mean(dim=2)


class VideoSwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: (B, T, H*W, C)
        b, t, _, c = tokens.shape
        orig_h, orig_w = h, w
        x = tokens.reshape(b, t, h, w, c)
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h = h + pad_h
            w = w + pad_w
        x = x.view(b, t, h // ws, ws, w // ws, ws, c)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, T, num_h, num_w, ws, ws, C)
        windows = x.reshape(b * t * (h // ws) * (w // ws), ws * ws, c)
        attn_in = self.norm1(windows)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))
        windows = windows.view(b, t, h // ws, w // ws, ws, ws, c)
        windows = windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, h, w, c)
        windows = windows[:, :, :orig_h, :orig_w, :]
        tokens = windows.reshape(b, t, orig_h * orig_w, c)
        return tokens


class VideoSwinBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_heads: int, patch_size: int, window_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            padding=(0, 0, 0),
        )
        self.blocks = nn.ModuleList([VideoSwinBlock(hidden_dim, num_heads, window_size=window_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.permute(0, 2, 1, 3, 4)
        x = self.proj(x)
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, c)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        tokens = self.norm(tokens)
        return tokens.mean(dim=2)


class TemporalCNNBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1, dilation=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, hidden_dim, kernel_size=3, stride=(1, 2, 2), padding=2, dilation=(2, 1, 1))
        self.norm1 = nn.BatchNorm3d(64)
        self.norm2 = nn.BatchNorm3d(128)
        self.norm3 = nn.BatchNorm3d(hidden_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.permute(0, 2, 1, 3, 4)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.mean(dim=[3, 4])  # spatial pooling, preserve time
        x = x.permute(0, 2, 1)
        return x


class SimpleVideoEncoder(nn.Module):
    """Configurable video encoder for conditioning embeddings.

    Supports TimeSformer / Video Swin / ViT-3D style transformer backbones as well as a
    stride-1 temporal CNN. All variants preserve the temporal dimension (no downsampling)
    and inject temporal positional encoding along with action-label embeddings into the
    final conditioning vector. The `temporal_unet` variant follows the requested multi-
    scale 2D conv + temporal FiLM/1D Conv design without temporal downsampling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        emb_dim: int = 256,
        encoder_type: str = "temporal_unet",
        num_layers: int = 2,
        num_heads: int = 4,
        patch_size: int = 4,
        window_size: int = 7,
        num_actions: Optional[int] = None,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type not in {"timesformer", "video_swin", "vit3d", "cnn", "temporal_unet"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.temporal_pos = TemporalPositionalEncoding(hidden_dim)
        self.num_actions = num_actions or len(ACTIONS)
        self.action_embed = nn.Embedding(self.num_actions, hidden_dim)
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)
        self.token_proj = nn.Linear(hidden_dim, emb_dim)

        if encoder_type == "timesformer":
            self.backbone = TimeSformerBackbone(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                patch_size=patch_size,
            )
        elif encoder_type == "video_swin":
            self.backbone = VideoSwinBackbone(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                patch_size=patch_size,
                window_size=window_size,
            )
        elif encoder_type == "vit3d":
            self.backbone = ViT3DBackbone(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                patch_size=patch_size,
            )
        elif encoder_type == "temporal_unet":
            self.backbone = TemporalUNetEncoder(
                in_channels=in_channels,
                cond_dim=emb_dim,
                base_channels=hidden_dim // 4,
                num_scales=3,
                num_actions=self.num_actions,
            )
        else:
            self.backbone = TemporalCNNBackbone(in_channels=in_channels, hidden_dim=hidden_dim)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, video: torch.Tensor, labels: Optional[torch.Tensor] = None) -> UNetConditioning:
        # video: (B, T, C, H, W)
        if self.encoder_type == "temporal_unet":
            # TemporalUNetEncoder already returns conditioning.
            cond = self.backbone(video, labels)
            return cond

        tokens = self.backbone(video)  # (B, T, hidden_dim)
        b, t, c = tokens.shape
        pos = self.temporal_pos(t, tokens.device).unsqueeze(0).unsqueeze(2)  # (1, T, 1, C)
        tokens = tokens + pos.squeeze(2)
        pooled = tokens.mean(dim=1)
        if labels is not None:
            label_emb = self.action_proj(self.action_embed(labels))
            pooled = pooled + label_emb
            tokens = tokens + label_emb.unsqueeze(1)
        pooled = self.final_norm(pooled)
        cond_vec = self.proj(pooled)
        cond_tokens = self.token_proj(tokens)
        scale_tokens = [cond_tokens for _ in range(3)]
        return UNetConditioning(vector=cond_vec, scale_tokens=scale_tokens)

    def encode_labels(self, labels: torch.Tensor) -> UNetConditioning:
        label_emb = self.action_proj(self.action_embed(labels))
        pooled = self.final_norm(label_emb)
        cond_vec = self.proj(pooled)
        return UNetConditioning(vector=cond_vec, scale_tokens=None)
