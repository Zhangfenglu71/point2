from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
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
    use_amp: bool = True
    radar_channels: int = 1
    cond_dim: int = 256
    channel_mults: tuple[int, ...] = (1, 2, 4)
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-3
    enable_cache: bool = True
    cache_in_workers: bool = False
    preload_videos: bool = False


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

        self.use_cond = cfg.exp in {"B_cond", "C_film", "C_full", "D_full"}
        cond_dim = cfg.cond_dim if self.use_cond else None
        self.model = UNet(
            in_channels=cfg.radar_channels,
            base_channels=64,
            cond_dim=cond_dim,
            use_film=cfg.use_film,
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

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        radar = batch["radar"].to(self.device)
        t = torch.rand(radar.size(0), device=self.device)
        noise = torch.randn_like(radar)
        x_t = (1 - t).view(-1, 1, 1, 1) * noise + t.view(-1, 1, 1, 1) * radar
        target_v = radar - noise

        cond_emb = None
        if self.use_cond and self.video_encoder is not None:
            video = batch["video"].to(self.device)
            cond_emb_full = self.video_encoder(video)
            drop_mask = (torch.rand(radar.size(0), device=self.device) < self.cfg.cond_drop).float()
            cond_mask = (1.0 - drop_mask).view(-1, 1)
            cond_emb = cond_emb_full * cond_mask

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            pred_v = self.model(x_t, t, cond_emb)
            loss = torch.mean((pred_v - target_v) ** 2)
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
