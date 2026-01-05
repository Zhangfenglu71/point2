from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.real_video_radar import ACTIONS, RealVideoRadarConfig, RealVideoRadarDataset
from engine.ckpt import save_checkpoint
from engine.seed import seed_all
from models.unet import UNet
from models.video_encoder import SimpleVideoEncoder


@dataclass
class TrainConfig:
    exp: str
    root: str
    split_train: str = "train"
    split_val: str = "val"
    img_size: int = 120
    clip_len: int = 64
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 0
    run_name: Optional[str] = None
    cond_drop: float = 0.25
    use_film: bool = False
    use_cross_attn: bool = False
    cross_heads: int = 4
    use_amp: bool = True
    radar_channels: int = 1
    cond_dim: int = 256
    channel_mults: tuple[int, ...] = (1, 2, 4)
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    enable_cache: bool = True
    cache_in_workers: bool = False
    preload_videos: bool = False
    freq_lambda: float = 0.0
    freq_band_split1: float = 1.0 / 3.0
    freq_band_split2: float = 2.0 / 3.0
    debug_freq: int = 0
    grad_lambda: float = 0.0
    grad_mode: str = "finite_diff"
    grad_on: str = "cond_only"
    debug_grad: int = 0
    taware: int = 0
    t_low: float = 0.3
    t_high: float = 0.7
    t_mix_power: float = 1.0


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        seed_all(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self._dataset_cache_enabled = cfg.enable_cache
        if cfg.num_workers > 0 and cfg.enable_cache and not cfg.cache_in_workers:
            self._dataset_cache_enabled = False
            print(
                "[Trainer] Disabling dataset cache because multiple DataLoader workers are enabled. "
                "Set --cache_in_workers=1 to force caching (may use significant RAM)."
            )
        # Fixed default run name per experiment for stable checkpoint paths.
        self.run_name = cfg.run_name or f"train_{cfg.exp}"
        self.run_dir = os.path.join("outputs", "runs", self.run_name)
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.ckpt_dir = os.path.join(self.run_dir, "ckpt")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # dataset
        train_dataset = RealVideoRadarDataset(
            RealVideoRadarConfig(
                root=cfg.root,
                split=cfg.split_train,
                img_size=cfg.img_size,
                clip_len=cfg.clip_len,
                radar_channels=cfg.radar_channels,
            ),
            seed=cfg.seed,
            enable_cache=self._dataset_cache_enabled,
            preload_videos=cfg.preload_videos,
        )
        val_dataset = RealVideoRadarDataset(
            RealVideoRadarConfig(
                root=cfg.root,
                split=cfg.split_val,
                img_size=cfg.img_size,
                clip_len=cfg.clip_len,
                radar_channels=cfg.radar_channels,
            ),
            seed=cfg.seed + 1,
            enable_cache=self._dataset_cache_enabled,
            preload_videos=cfg.preload_videos,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        self.use_cond = cfg.exp in {"B_cond", "C_film", "D_full", "E_full", "F_freq", "G_grad", "H_taware"}
        cond_dim = cfg.cond_dim if self.use_cond else None
        self.model = UNet(
            in_channels=cfg.radar_channels,
            base_channels=64,
            cond_dim=cond_dim,
            use_film=cfg.use_film,
            use_cross_attn=cfg.use_cross_attn,
            cross_heads=cfg.cross_heads,
            channel_mults=cfg.channel_mults,
        ).to(self.device)
        if self.use_cond:
            self.video_encoder = SimpleVideoEncoder(emb_dim=cond_dim).to(self.device)
        else:
            self.video_encoder = None

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + (list(self.video_encoder.parameters()) if self.video_encoder else []),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        scaler_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_enabled = cfg.use_amp and scaler_device == "cuda"
        self.scaler = torch.amp.GradScaler(scaler_device, enabled=self.amp_enabled)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_no_significant_improve = 0
        self._save_config()

        self.freq_band_edges: Optional[tuple[int, int, int, int]] = None
        self.freq_energy_proportions: Optional[tuple[float, float, float]] = None
        if self.cfg.exp in {"F_freq", "H_taware"} and self.cfg.freq_lambda > 0:
            self._init_frequency_bands()

    def _init_frequency_bands(self, max_batches: int = 64) -> None:
        """
        Estimate low/mid/high band boundaries from training data before epoch 1.
        """
        total_energy: Optional[Tensor] = None
        total_count = 0
        batches_used = 0

        with torch.no_grad():
            for batch in self.train_loader:
                radar = batch["radar"].to(self.device).float()  # (B, C, H, W)
                energy = torch.abs(radar)
                energy_per_sample = energy.mean(dim=(1, 3))  # (B, H)

                if total_energy is None:
                    total_energy = torch.zeros_like(energy_per_sample[0])
                total_energy += energy_per_sample.sum(dim=0)
                total_count += energy_per_sample.size(0)
                batches_used += 1
                if batches_used >= max_batches:
                    break

        if total_energy is None or total_count == 0:
            print("[F_freq] Warning: could not estimate frequency bands (no data); disabling freq loss.")
            self.freq_band_edges = None
            self.freq_energy_proportions = None
            return

        mean_energy = total_energy / total_count  # (H,)
        cdf = torch.cumsum(mean_energy, dim=0)
        total_energy_sum = cdf[-1].item()
        if total_energy_sum <= 0:
            print("[F_freq] Warning: non-positive energy detected; disabling freq loss.")
            self.freq_band_edges = None
            self.freq_energy_proportions = None
            return

        freq_len = mean_energy.shape[0]
        thresholds = torch.tensor([0.33, 0.66], device=mean_energy.device) * total_energy_sum
        i_low = int(torch.searchsorted(cdf, thresholds[0], right=False).item())
        i_high = int(torch.searchsorted(cdf, thresholds[1], right=False).item())

        i_low = min(max(i_low, 1), freq_len - 2)
        i_high = min(max(i_high, i_low + 1), freq_len - 1)
        self.freq_band_edges = (0, i_low, i_high, freq_len)

        energy_low = mean_energy[:i_low].sum().item()
        energy_mid = mean_energy[i_low:i_high].sum().item()
        energy_high = mean_energy[i_high:].sum().item()
        self.freq_energy_proportions = (
            energy_low / total_energy_sum,
            energy_mid / total_energy_sum,
            energy_high / total_energy_sum,
        )

        print(
            "[F_freq] Estimated frequency bands "
            f"(freq_len={freq_len}, batches={batches_used}, samples={total_count}): "
            f"i_low={i_low}, i_high={i_high}, "
            f"ratios=({i_low / freq_len:.4f}, {i_high / freq_len:.4f}), "
            f"energy_ratio="
            f"({self.freq_energy_proportions[0]:.4f}, "
            f"{self.freq_energy_proportions[1]:.4f}, "
            f"{self.freq_energy_proportions[2]:.4f})"
        )

    def _extra_state(self) -> Dict[str, Any]:
        return {"video_encoder": self.video_encoder.state_dict() if self.video_encoder else {}}

    def _save_config(self) -> None:
        git_state = "dirty"
        try:
            status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
            git_state = "clean" if status == "" else "dirty"
        except Exception:
            git_state = "unavailable"
        cfg = asdict(self.cfg)
        cfg["git_state"] = git_state
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def _frequency_band_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        cond_mask: torch.Tensor,
        per_sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_mask = cond_mask.view(-1)
        if per_sample_weight is not None:
            per_sample_weight = per_sample_weight.view(-1)
            effective = cond_mask * per_sample_weight
        else:
            effective = cond_mask
        if not torch.any(effective > 0):
            return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
        if self.freq_band_edges is None:
            return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

        x_pred = x_pred.float()
        x_gt = x_gt.float()
        energy_pred = torch.abs(x_pred)
        energy_gt = torch.abs(x_gt)

        band_edges = self.freq_band_edges

        band_losses = []
        for start, end in zip(band_edges[:-1], band_edges[1:]):
            if end <= start:
                continue
            pred_band = energy_pred[:, :, start:end, :]
            gt_band = energy_gt[:, :, start:end, :]
            pred_stat = pred_band.mean(dim=(-2, -1))
            gt_stat = gt_band.mean(dim=(-2, -1))
            band_losses.append(torch.abs(pred_stat - gt_stat))

        if not band_losses:
            return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

        stacked = torch.stack(band_losses, dim=0).mean(dim=0)  # (B, C)
        per_sample_loss = stacked.mean(dim=1)
        weighted = (per_sample_loss * effective).sum() / effective.sum()
        return weighted

    def _spectral_gradient_loss(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        cond_mask: torch.Tensor,
        per_sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_mask = cond_mask.view(-1)
        if per_sample_weight is not None:
            per_sample_weight = per_sample_weight.view(-1)
            effective = cond_mask * per_sample_weight
        else:
            effective = cond_mask
        if not torch.any(effective > 0):
            return torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)

        x_pred = x_pred.float()
        x_gt = x_gt.float()

        grad_t_pred = x_pred[..., :, 1:] - x_pred[..., :, :-1]
        grad_t_gt = x_gt[..., :, 1:] - x_gt[..., :, :-1]
        grad_f_pred = x_pred[..., 1:, :] - x_pred[..., :-1, :]
        grad_f_gt = x_gt[..., 1:, :] - x_gt[..., :-1, :]

        stat_t_pred = grad_t_pred.abs().mean(dim=(-2, -1))
        stat_t_gt = grad_t_gt.abs().mean(dim=(-2, -1))
        stat_f_pred = grad_f_pred.abs().mean(dim=(-2, -1))
        stat_f_gt = grad_f_gt.abs().mean(dim=(-2, -1))

        per_sample_loss = (torch.abs(stat_t_pred - stat_t_gt) + torch.abs(stat_f_pred - stat_f_gt)).mean(dim=1)
        weighted = (per_sample_loss * effective).sum() / effective.sum()
        return weighted.to(dtype=x_gt.dtype)

    def _taware_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.t_high <= self.cfg.t_low:
            raise ValueError(f"t_high ({self.cfg.t_high}) must be greater than t_low ({self.cfg.t_low}) for t-aware loss")
        denom = max(self.cfg.t_high - self.cfg.t_low, 1e-6)
        u = ((t - self.cfg.t_low) / denom).clamp(0.0, 1.0)
        if self.cfg.t_mix_power != 1.0:
            u = torch.pow(u, self.cfg.t_mix_power)
        w_grad = u
        w_freq = 1.0 - u
        return w_freq, w_grad

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        radar = batch["radar"].to(self.device)
        t = torch.rand(radar.size(0), device=self.device)
        noise = torch.randn_like(radar)
        x_t = (1 - t).view(-1, 1, 1, 1) * noise + t.view(-1, 1, 1, 1) * radar
        target_v = radar - noise

        cond_emb = None
        cond_mask = None
        if self.use_cond and self.video_encoder is not None:
            video = batch["video"].to(self.device)
            cond_emb_full = self.video_encoder(video)
            drop_mask = (torch.rand(radar.size(0), device=self.device) < self.cfg.cond_drop).float()
            cond_mask = (1.0 - drop_mask).view(-1, 1)
            cond_emb = cond_emb_full * cond_mask

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            pred_v = self.model(x_t, t, cond_emb)
            main_loss = torch.mean((pred_v - target_v) ** 2)
            freq_loss = torch.zeros((), device=self.device, dtype=pred_v.dtype)
            grad_loss = torch.zeros((), device=self.device, dtype=pred_v.dtype)
            x_hat = x_t + (1.0 - t).view(-1, 1, 1, 1) * pred_v
            w_freq: Optional[torch.Tensor] = None
            w_grad: Optional[torch.Tensor] = None
            grad_mask = cond_mask
            if grad_mask is not None and self.cfg.grad_on == "all":
                grad_mask = torch.ones_like(grad_mask)
            if self.cfg.exp == "H_taware" and self.cfg.taware:
                w_freq, w_grad = self._taware_weights(t)
            if self.cfg.freq_lambda > 0 and cond_mask is not None and torch.any(cond_mask > 0):
                freq_loss = self._frequency_band_loss(x_hat, radar, cond_mask, per_sample_weight=w_freq)
                if self.cfg.debug_freq and self.global_step % max(self.cfg.debug_freq, 1) == 0:
                    mask = cond_mask.view(-1) > 0
                    if torch.any(mask):
                        x_hat_masked = x_hat[mask]
                        radar_masked = radar[mask]
                        x_stats = (
                            x_hat_masked.min().item(),
                            x_hat_masked.max().item(),
                            x_hat_masked.mean().item(),
                            x_hat_masked.std().item(),
                        )
                        r_stats = (
                            radar_masked.min().item(),
                            radar_masked.max().item(),
                            radar_masked.mean().item(),
                            radar_masked.std().item(),
                        )
                        print(
                            "[F_freq][debug] step "
                            f"{self.global_step}: x_hat min/max/mean/std="
                            f"{x_stats[0]:.2f}/{x_stats[1]:.2f}/{x_stats[2]:.2f}/{x_stats[3]:.2f}, "
                            f"radar min/max/mean/std="
                            f"{r_stats[0]:.2f}/{r_stats[1]:.2f}/{r_stats[2]:.2f}/{r_stats[3]:.2f}, "
                            f"freq_loss={freq_loss.item():.6f}"
                        )
            if self.cfg.grad_lambda > 0 and grad_mask is not None and torch.any(grad_mask > 0):
                grad_loss = self._spectral_gradient_loss(x_hat, radar, grad_mask, per_sample_weight=w_grad)
                if self.cfg.debug_grad and self.global_step % max(self.cfg.debug_grad, 1) == 0:
                    mask = grad_mask.view(-1) > 0
                    if torch.any(mask):
                        x_hat_masked = x_hat[mask]
                        radar_masked = radar[mask]
                        x_stats = (
                            x_hat_masked.min().item(),
                            x_hat_masked.max().item(),
                            x_hat_masked.mean().item(),
                            x_hat_masked.std().item(),
                        )
                        r_stats = (
                            radar_masked.min().item(),
                            radar_masked.max().item(),
                            radar_masked.mean().item(),
                            radar_masked.std().item(),
                        )
                        print(
                            "[G_grad][debug] step "
                            f"{self.global_step}: x_hat min/max/mean/std="
                            f"{x_stats[0]:.2f}/{x_stats[1]:.2f}/{x_stats[2]:.2f}/{x_stats[3]:.2f}, "
                            f"radar min/max/mean/std="
                            f"{r_stats[0]:.2f}/{r_stats[1]:.2f}/{r_stats[2]:.2f}/{r_stats[3]:.2f}, "
                            f"grad_loss={grad_loss.item():.6f}"
                        )
            loss = main_loss + self.cfg.freq_lambda * freq_loss + self.cfg.grad_lambda * grad_loss
        return loss

    def _run_epoch(self, epoch: int, train: bool = True) -> float:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        if self.video_encoder:
            self.video_encoder.train(train)
        total_loss = 0.0
        progress_desc = f"{'train' if train else 'val'} epoch {epoch}"
        for batch in tqdm(loader, desc=progress_desc, dynamic_ncols=True):
            if train:
                self.optimizer.zero_grad()
                loss = self._forward(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1
            else:
                with torch.no_grad():
                    loss = self._forward(batch)
            total_loss += loss.item() * batch["radar"].size(0)
        return total_loss / len(loader.dataset)

    def _save_best_metrics(self, epoch: int, train_loss: float, val_loss: float) -> None:
        best_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        best_path = os.path.join(self.log_dir, "best_metrics.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

    def fit(self) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"Starting epoch {epoch}/{self.cfg.epochs} (best val loss: {self.best_val_loss:.4f})")
            train_loss = self._run_epoch(epoch, train=True)
            train_metrics = {"train_loss": train_loss}
            save_checkpoint(
                os.path.join(self.ckpt_dir, "last.ckpt"),
                self.model,
                self.optimizer,
                epoch,
                self.global_step,
                config=asdict(self.cfg),
                metrics=train_metrics,
                extra_state=self._extra_state(),
                save_last=True,
            )
            val_loss = self._run_epoch(epoch, train=False)
            improvement_over_best = self.best_val_loss - val_loss
            is_best = val_loss < self.best_val_loss
            significant_improvement = improvement_over_best > self.cfg.early_stop_min_delta or self.best_val_loss == float(
                "inf"
            )

            if is_best:
                self.best_val_loss = val_loss
                self._save_best_metrics(epoch, train_loss, val_loss)
                print(f"New best val loss: {val_loss:.4f} at epoch {epoch}")
            if significant_improvement:
                self.epochs_no_significant_improve = 0
            else:
                self.epochs_no_significant_improve += 1
            print(
                f"Epoch {epoch}/{self.cfg.epochs}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} best_val_loss={self.best_val_loss:.4f}"
            )
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.ckpt")
            save_checkpoint(
                ckpt_path,
                self.model,
                self.optimizer,
                epoch,
                self.global_step,
                config=asdict(self.cfg),
                best=is_best,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
                extra_state=self._extra_state(),
                save_last=True,
            )

            if self.cfg.early_stop_patience > 0 and self.epochs_no_significant_improve >= self.cfg.early_stop_patience:
                print(
                    "Early stopping triggered: "
                    f"no significant val loss improvement for {self.cfg.early_stop_patience} epochs "
                    f"(min_delta={self.cfg.early_stop_min_delta})."
                )
                break


def run_training(cfg: TrainConfig) -> None:
    trainer = Trainer(cfg)
    trainer.fit()


__all__ = ["TrainConfig", "run_training"]
