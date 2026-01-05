from __future__ import annotations

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _gaussian_kernel(window_size: int, sigma: float, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    return kernel


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """Structural similarity index for 4D tensors (B, C, H, W)."""

    if x.shape != y.shape:
        raise ValueError(f"SSIM input shapes must match, got {x.shape} and {y.shape}")

    channel = x.size(1)
    kernel = _gaussian_kernel(window_size, sigma, channel, x.device, x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channel)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channel)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channel) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + eps)
    return ssim_map.mean()


class _VGGFeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[int]) -> None:
        super().__init__()
        try:
            weights = models.VGG16_Weights.IMAGENET1K_FEATURES
        except Exception:
            weights = None
        vgg16 = models.vgg16(weights=weights).features
        self.layers = set(layers)
        self.vgg_slice = nn.Sequential(*[layer for _, layer in vgg16._modules.items() if layer is not None])
        for param in self.vgg_slice.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        h = x
        for idx, layer in enumerate(self.vgg_slice):
            h = layer(h)
            if idx in self.layers:
                feats.append(h)
        return feats


class PerceptualLoss(nn.Module):
    def __init__(self, kind: str = "vgg", layers: Iterable[int] = (3, 8, 15, 22)) -> None:
        super().__init__()
        self.kind = kind.lower()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Default to VGG-based perceptual features; attempt AudioSet (VGGish) when requested.
        if self.kind == "audioset":
            try:
                self.backbone = torch.hub.load("harritaylor/torchvggish", "vggish")
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()
                return
            except Exception:
                # Fall back to VGG features if hub loading fails.
                print("[PerceptualLoss] Falling back to VGG16 features (AudioSet backend unavailable).")

        self.backbone = _VGGFeatureExtractor(layers)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x.clamp(-1, 1) + 1) / 2.0  # to [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_p = self._preprocess(x)
        y_p = self._preprocess(y)

        if isinstance(self.backbone, _VGGFeatureExtractor):
            feats_x = self.backbone(x_p)
            feats_y = self.backbone(y_p)
        else:
            # VGGish expects mono log-mel inputs; approximate by average over spatial dims.
            x_mel = x_p.mean(dim=[2, 3])
            y_mel = y_p.mean(dim=[2, 3])
            feats_x = [self.backbone(x_mel)]
            feats_y = [self.backbone(y_mel)]

        total = 0.0
        for fx, fy in zip(feats_x, feats_y):
            total = total + F.l1_loss(fx, fy)
        return total / max(1, len(feats_x))


def band_l1_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    band_edges: Optional[tuple[int, int, int, int]],
    cond_mask: Optional[torch.Tensor],
    per_sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if band_edges is None:
        return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

    cond_mask = cond_mask.view(-1) if cond_mask is not None else torch.ones(x_pred.size(0), device=x_pred.device)
    if per_sample_weight is not None:
        per_sample_weight = per_sample_weight.view(-1)
        effective = cond_mask * per_sample_weight
    else:
        effective = cond_mask
    if not torch.any(effective > 0):
        return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

    diffs = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        if end <= start:
            continue
        band_pred = x_pred[:, :, start:end, :]
        band_gt = x_gt[:, :, start:end, :]
        diffs.append(torch.abs(band_pred - band_gt).mean(dim=(1, 2, 3)))

    if not diffs:
        return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

    stacked = torch.stack(diffs, dim=1).mean(dim=1)
    weighted = (stacked * effective).sum() / effective.sum()
    return weighted


def temporal_smoothness_loss(x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    if x_pred.size(-1) < 2:
        return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
    diff_pred = x_pred[..., 1:] - x_pred[..., :-1]
    diff_gt = x_gt[..., 1:] - x_gt[..., :-1]
    return F.l1_loss(diff_pred, diff_gt)


def info_nce_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if emb.ndim != 2:
        emb = emb.flatten(1)
    emb = F.normalize(emb, dim=1)
    batch_size = emb.size(0)
    if batch_size <= 1:
        return torch.zeros((), device=emb.device, dtype=emb.dtype)

    sim = emb @ emb.t() / temperature
    mask = torch.eye(batch_size, device=emb.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -math.inf)

    labels = labels.view(-1)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & (~mask)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Avoid log(0) when no positives exist for a sample.
    pos_exp = exp_sim * positive_mask
    pos_sum = pos_exp.sum(dim=1)
    valid = pos_sum > 0
    if not torch.any(valid):
        return torch.zeros((), device=emb.device, dtype=emb.dtype)

    log_prob = torch.zeros_like(pos_sum)
    log_prob[valid] = torch.log(pos_sum[valid] / denom[valid, 0])
    return -log_prob[valid].mean()

